#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__
#include <libproc.h>
#include <sys/proc.h>
#include <sys/proc_info.h>
#elif defined(__linux__)
#include <dirent.h>
#else
#error "process-tree-snapshot supports only macOS and Linux"
#endif

struct process_snapshot {
  long pid;
  long parent_pid;
  long process_group;
  int64_t resident_kib;
  uint64_t start_high;
  uint64_t start_low;
  char state;
};

#ifdef __APPLE__

static int read_process(long pid, struct process_snapshot *snapshot) {
  struct proc_bsdinfo bsd_info;
  int bsd_size = proc_pidinfo((int)pid, PROC_PIDTBSDINFO, 0, &bsd_info,
                              (int)sizeof(bsd_info));
  if (bsd_size != (int)sizeof(bsd_info)) {
    return 0;
  }

  struct proc_taskinfo task_info;
  int task_size = proc_pidinfo((int)pid, PROC_PIDTASKINFO, 0, &task_info,
                               (int)sizeof(task_info));
  int64_t resident_kib = -1;
  if (task_size == (int)sizeof(task_info)) {
    resident_kib = (int64_t)(task_info.pti_resident_size / 1024U);
  } else {
    struct proc_bsdinfo still_present;
    int retry_size = proc_pidinfo((int)pid, PROC_PIDTBSDINFO, 0, &still_present,
                                  (int)sizeof(still_present));
    if (retry_size != (int)sizeof(still_present)) {
      return 0;
    }
  }

  snapshot->pid = (long)bsd_info.pbi_pid;
  snapshot->parent_pid = (long)bsd_info.pbi_ppid;
  snapshot->process_group = (long)bsd_info.pbi_pgid;
  snapshot->resident_kib = resident_kib;
  snapshot->start_high = (uint64_t)bsd_info.pbi_start_tvsec;
  snapshot->start_low = (uint64_t)bsd_info.pbi_start_tvusec;
  snapshot->state = bsd_info.pbi_status == SZOMB ? 'Z' : 'A';
  return 1;
}

static int emit_all_processes(void) {
  int estimate = proc_listallpids(NULL, 0);
  if (estimate <= 0) {
    fprintf(stderr, "cannot enumerate processes\n");
    return 1;
  }

  size_t capacity = (size_t)estimate + 128U;
  pid_t *pids = NULL;
  int count = 0;
  for (;;) {
    pid_t *next = realloc(pids, capacity * sizeof(*pids));
    if (next == NULL) {
      free(pids);
      fprintf(stderr, "cannot allocate process list\n");
      return 1;
    }
    pids = next;
    count = proc_listallpids(pids, (int)(capacity * sizeof(*pids)));
    if (count < 0) {
      free(pids);
      fprintf(stderr, "cannot enumerate processes\n");
      return 1;
    }
    if ((size_t)count < capacity) {
      break;
    }
    capacity *= 2U;
  }

  for (int index = 0; index < count; ++index) {
    if (pids[index] <= 0) {
      continue;
    }
    struct process_snapshot snapshot;
    if (!read_process((long)pids[index], &snapshot)) {
      continue;
    }
    printf("%ld %ld %ld %" PRId64 " %" PRIu64 ":%" PRIu64 " %c\n",
           snapshot.pid, snapshot.parent_pid, snapshot.process_group,
           snapshot.resident_kib, snapshot.start_high, snapshot.start_low,
           snapshot.state);
  }
  free(pids);
  return ferror(stdout) ? 1 : 0;
}

#elif defined(__linux__)

static int parse_pid_name(const char *name, long *pid) {
  if (*name == '\0') {
    return 0;
  }
  for (const unsigned char *cursor = (const unsigned char *)name;
       *cursor != '\0'; ++cursor) {
    if (!isdigit(*cursor)) {
      return 0;
    }
  }
  errno = 0;
  char *end = NULL;
  long parsed = strtol(name, &end, 10);
  if (errno != 0 || end == name || *end != '\0' || parsed <= 0) {
    return 0;
  }
  *pid = parsed;
  return 1;
}

static int read_process(long pid, struct process_snapshot *snapshot) {
  char path[64];
  int path_length = snprintf(path, sizeof(path), "/proc/%ld/stat", pid);
  if (path_length < 0 || (size_t)path_length >= sizeof(path)) {
    return 0;
  }
  FILE *input = fopen(path, "r");
  if (input == NULL) {
    return 0;
  }
  char line[4096];
  if (fgets(line, sizeof(line), input) == NULL) {
    fclose(input);
    return 0;
  }
  if (fclose(input) != 0) {
    return 0;
  }
  char *right_parenthesis = strrchr(line, ')');
  if (right_parenthesis == NULL || right_parenthesis[1] != ' ') {
    return 0;
  }

  char state = '\0';
  long parent_pid = 0;
  long process_group = 0;
  unsigned long long start_ticks = 0;
  long resident_pages = 0;
  int matched = sscanf(
      right_parenthesis + 2,
      "%c %ld %ld %*d %*d %*d %*u %*lu %*lu %*lu %*lu %*lu %*lu "
      "%*ld %*ld %*ld %*ld %*ld %*ld %llu %*lu %ld",
      &state, &parent_pid, &process_group, &start_ticks, &resident_pages);
  if (matched != 5) {
    return 0;
  }

  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || resident_pages < 0) {
    return 0;
  }
  snapshot->pid = pid;
  snapshot->parent_pid = parent_pid;
  snapshot->process_group = process_group;
  snapshot->resident_kib =
      (int64_t)resident_pages * (int64_t)page_size / 1024;
  snapshot->start_high = 0;
  snapshot->start_low = (uint64_t)start_ticks;
  snapshot->state = state == 'Z' ? 'Z' : 'A';
  return 1;
}

static int emit_all_processes(void) {
  DIR *directory = opendir("/proc");
  if (directory == NULL) {
    fprintf(stderr, "cannot enumerate processes\n");
    return 1;
  }
  struct dirent *entry;
  while ((entry = readdir(directory)) != NULL) {
    long pid = 0;
    if (!parse_pid_name(entry->d_name, &pid)) {
      continue;
    }
    struct process_snapshot snapshot;
    if (!read_process(pid, &snapshot)) {
      continue;
    }
    printf("%ld %ld %ld %" PRId64 " %" PRIu64 ":%" PRIu64 " %c\n",
           snapshot.pid, snapshot.parent_pid, snapshot.process_group,
           snapshot.resident_kib, snapshot.start_high, snapshot.start_low,
           snapshot.state);
  }
  int close_status = closedir(directory);
  return close_status == 0 && !ferror(stdout) ? 0 : 1;
}

#endif

static int emit_state(long pid) {
  struct process_snapshot snapshot;
  if (!read_process(pid, &snapshot)) {
    return 1;
  }
  printf("%c\n", snapshot.state);
  return ferror(stdout) ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    return emit_all_processes();
  }
  if (argc == 3 && strcmp(argv[1], "--state") == 0) {
    errno = 0;
    char *end = NULL;
    long pid = strtol(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || pid <= 0) {
      fprintf(stderr, "invalid pid\n");
      return 2;
    }
    return emit_state(pid);
  }
  fprintf(stderr, "usage: %s [--state PID]\n", argv[0]);
  return 2;
}
