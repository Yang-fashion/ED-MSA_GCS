import os
import shutil
import torch
import glob
import json
import time


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('runs', args.train_dataset, args.arch)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'exp_*')))
        run_id = time.strftime("%Y-%m-%d_%H:%M", time.localtime())

        self.experiment_dir = os.path.join(self.directory, 'exp_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # Save args
        with open(os.path.join(self.experiment_dir, 'args_parameters.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        # shutil.copyfile(filename, os.path.join(self.directory, 'model_last.pth'))
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        try:
                            with open(path, 'r') as f:
                                miou = float(f.readline())
                                previous_miou.append(miou)
                        except:
                            pass
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))

    def save_experiment_config(self):
        # Save args
        logfile = os.path.join(self.experiment_dir, 'args_parameters.txt')
        log_file = open(logfile, 'w')
        args = self.args
        with log_file as f:
            json.dump(args.__dict__, f, indent=2)

        log_file.close()
