import time
import os
import subprocess

PLOT_TIME_INTERVAL = 0.5


def auto_demo():
    current_microbatch_no = 0
    first_time_read = 1
    while current_microbatch_no < 950:
        time.sleep(PLOT_TIME_INTERVAL)
        print(current_microbatch_no)
        if os.path.exists("temp_mcid_record.txt"):
            with open("temp_mcid_record.txt", 'r') as f:
                current_microbatch_no = int(f.readlines()[-1])
        if current_microbatch_no == 100 and first_time_read == 1:
            subprocess.Popen(['sudo', 'tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'tbf', 'rate', '500Mbit', 'latency', '50ms', 'burst', '15kb'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            first_time_read == 0
        elif current_microbatch_no == 299:
            first_time_read == 1
        elif current_microbatch_no == 300 and first_time_read == 1:
            subprocess.Popen(['sudo', 'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.Popen(['sudo', 'tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'tbf', 'rate', '50Mbit', 'latency', '50ms', 'burst', '15kb'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            first_time_read == 0
        elif current_microbatch_no == 499:
            first_time_read == 1
        elif current_microbatch_no == 500 and first_time_read == 1:
            subprocess.Popen(['sudo', 'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.Popen(['sudo', 'tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'tbf', 'rate', '250Mbit', 'latency', '50ms', 'burst', '15kb'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            first_time_read == 0
        elif current_microbatch_no == 699:
            first_time_read == 1
        elif current_microbatch_no == 700 and first_time_read == 1:
            subprocess.Popen(['sudo', 'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            first_time_read == 0
    return

if __name__ == '__main__':
    auto_demo()