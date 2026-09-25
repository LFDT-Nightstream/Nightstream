#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#include <stdio.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 1;
        fprintf(stderr, "device=%s allocated=%lu\n", device.name.UTF8String, (unsigned long)device.currentAllocatedSize);
        id<MTLCommandQueue> queue = [device newCommandQueueWithMaxCommandBufferCount:1];
        id<MTLBuffer> buffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        command.label = @"nightstream isolated buffer fill";
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [command addScheduledHandler:^(id<MTLCommandBuffer> current) {
            fprintf(stderr, "scheduled status=%lu\n", (unsigned long)current.status);
        }];
        [command addCompletedHandler:^(id<MTLCommandBuffer> current) {
            fprintf(stderr, "completed status=%lu error=%s\n", (unsigned long)current.status,
                current.error ? current.error.description.UTF8String : "none");
            dispatch_semaphore_signal(done);
        }];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        [encoder fillBuffer:buffer range:NSMakeRange(0, sizeof(uint32_t)) value:0x5a];
        [encoder endEncoding];
        [command commit];
        fprintf(stderr, "committed status=%lu allocated=%lu\n", (unsigned long)command.status,
            (unsigned long)device.currentAllocatedSize);
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        return command.status == MTLCommandBufferStatusCompleted && *(uint32_t *)buffer.contents == 0x5a5a5a5a ? 0 : 1;
    }
}
