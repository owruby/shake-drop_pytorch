# -*- coding: utf-8 -*

import os
import math
import json
from datetime import datetime


def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(t.data.view_as(pred)).cpu().sum()
    return acc


class Logger:

    def __init__(self, log_dir, headers):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, "log.txt"), "w")
        header_str = "\t".join(headers + ["EndTime."])
        self.print_str = "\t".join(["{}"] + ["{:.6f}"] * (len(headers) - 1) + ["{}"])

        self.f.write(header_str + "\n")
        self.f.flush()
        print(header_str)

    def write(self, *args):
        now_time = datetime.now().strftime("%m/%d %H:%M:%S")
        self.f.write(self.print_str.format(*args, now_time) + "\n")
        self.f.flush()
        print(self.print_str.format(*args, now_time))

    def write_hp(self, hp):
        json.dump(hp, open(os.path.join(self.log_dir, "hp.json"), "w"))

    def close(self):
        self.f.close()
