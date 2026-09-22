#include <libproc.h>
#include <stdint.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <string.h>
int memory_usage(int pid, uint64_t values[3]) {
    struct rusage_info_v6 usage;
    int result = proc_pid_rusage(pid, RUSAGE_INFO_V6, (rusage_info_t *)&usage);
    if (result == 0) {
        values[0] = usage.ri_resident_size;
        values[1] = usage.ri_phys_footprint;
        values[2] = usage.ri_lifetime_max_phys_footprint;
    }
    return result;
}
int child_exited(int pid, int block) {
    siginfo_t info;
    memset(&info, 0, sizeof(info));
    int flags = WEXITED | WNOWAIT | (block ? 0 : WNOHANG);
    if (waitid(P_PID, pid, &info, flags) != 0) return -1;
    return info.si_pid == pid;
}
