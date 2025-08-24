""" This script serves to track the resourses to correctly set server requirements. """
import time
import psutil
import threading
import os
import logging

logging.basicConfig(level=logging.INFO)

def start_resource_logger(pid=None, interval=1.0):
    pid = pid or os.getpid()
    p = psutil.Process(pid)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            with p.oneshot():
                rss = p.memory_info().rss / (1024*1024)  # MiB
                cpu = p.cpu_percent(interval=None)      # % od minulé výzvy
            logging.info(f"[RES] RSS={rss:.1f} MiB  CPU={cpu:5.1f}%")
            time.sleep(interval)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop