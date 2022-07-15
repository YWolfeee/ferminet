"""""""""""""""""""""""""""""""""
    `sweep.py` by YWolfeee (Jul 15)

    ## Decription ##
    This function serves as a script to run multiple commands together and gather statistics.
    You need to write a basic command in a file, input changable args using the parser below. 
    This function will append your changable args input the basic command per run.
    ##

    ## Attention ##
    The dir structure is as follows. You are required to input `base_dir`, where the scripts is loaded. In addition, based on your tag and changable args, the function will automaitcally build the `save_path` args before each run. The format will be like `{base_dir}/{tag}_{changable_args}/`.
    ##

    ## Running Method ##
    Notice that in order to make debugging easy, we don't use python module to run ferminet. Instead, standard python main.py command should be used.
    ##
"""""""""""""""""""""""""""""""""
import os 
import argparse
import time
import numpy as np
from copy import deepcopy
from typing import Tuple
from multiprocessing import Process, Value, Lock


def getArgs():
    parser = argparse.ArgumentParser()

    # loading and saving
    parser.add_argument("--base_dir", type = str, default = "working",
                        help = "The base dir to load scripts and to store resulting statistics. If must be created before calling this function, since we will load script from it.")
    parser.add_argument("--script", type = str, default = "script.sh", 
                        help = "The file that running command should be read from. It should only contain one command. If it has only one line, it is like `python main.py --config /path/to/h2_config.py`. It can also contain a `#!/bin/bash` at the beginning.")
    parser.add_argument("--tag", type = str, default = None,
                        help = "The prefixed added before each directory. This is to help you specify different settings (but not changable args). For example, you can use `C2H4` and `Zn` to represent different systems, since we don't regard it as a changable args.")
    parser.add_argument("--logging", type = str, default = "output.log",
                        help="Where the standard output will be saved. Notice that when logging is not None, you wouln't see any output from the ferminet in terminal, since they are redirect to a file. If you want to see them in the terminal, please set logging to 'None'.")
    parser.add_argument("--per_gpu", type = int, default = 1,
                        help = "Number of GPU each program should obtain. Notice that `per_gpu` should divide the length of `gpu_list`.")
    parser.add_argument("--gpu_list", nargs = "+", type = int, default = [0],
                        help = "List of gpu index that we can use.")

    # changable args. We will loop over these args. Once appeared, it will overwrite the ferminet config.
    """
        To add a new changable args, you can directly add an argument here. The function will automatically loop over them.
        For example, if you want to loop over lr rate, you can add `--loop.optim.lr.rate`, set nargs to `+` and type to float.
    """

    parser.add_argument("--loop.batch_size", nargs = "+", type = int, default = [128, 256],
                        help = "The batch_size list you want to loop over. Format should be like `--loop.batch_size 256 512 1024 2048 4096`.")
    parser.add_argument("--loop.optim.iterations", nargs = "+", type = int, default = [10, 20],
                        help = "Number of optimization steps.")

    args = parser.parse_args()

    # Do extra checking works here.
    if args.logging in ["none", "None"]:
        args.logging = None
    args.gpu_list = [str(w) for w in args.gpu_list]

    # Print arguments before returning.
    key_list = list(vars(args).keys())
    normal_key = [w for w in key_list if 'loop' not in w]
    loop_key = [w for w in key_list if 'loop' in w]
     
    print("-------- basic args --------")
    for key in normal_key:
        print("{}: {}".format(key, vars(args)[key]))
    print("-------- basic args --------")
    print("")
    print("-------- looping args --------")
    for key in loop_key:
        print("{}: {}".format(key[5:], vars(args)[key]))
    print("-------- looping args --------")
    print("")

    return args, loop_key


def convert_idx_to_args(idx: int, loop_dict: dict) -> dict:
    """Convert the index back into its corresponding args dict

    Args:
        idx (int): The index of this args. Falls in [0, total_runs).
        loop_list (list): The list of key that should be iterated.
        loop_dict (dict): The dict of key, each value being its candidate.

    Returns:
        dict: The corresponding dict of this index.
    """
    idx = deepcopy(idx)
    num_list = [len(values) for values in loop_dict.values()]
    dic, lis = {}, list(loop_dict.keys())
    for key, num in zip(lis[::-1], num_list[::-1]):
        idx, remain = idx // num, idx % num
        dic[key] = loop_dict[key][remain]
    return dic


def generate_command(
        args: argparse.Namespace,
        idx: int,
        loop_dict: dict,
        raw_cmd: str,
        gpu_list: list) -> Tuple[str, str]:
    """generate complete command according to idx and args.

    Args:
        args (argparse.Namespace): The original input args
        idx (int): the index of this command.
        loop_dict (dict): The dictionary of loop variables.
        raw_cmd (str): The raw command written in script.sh.
        gpu_list (list): List of gpu used by this worker.
    Returns:
        cmd (str): The complete command to be executed.
        save_path (str): The directory corresponding to this command.
    """
    append_args = convert_idx_to_args(idx, loop_dict)

    # Construct the command
    cmd = " ".join([
        "CUDA_VISIBLE_DEVICES={}".format(
            ",".join(gpu_list)),
        raw_cmd,
        *["--config.{} {}".format(
                key, append_args[key]
            ) for key in list(loop_dict.keys())]
    ])

    # Append save_path
    subdir = "+".join(["{}_{}".format(
            key, append_args[key]
        ) for key in list(loop_dict.keys())])
    if args.tag is not None:
        subdir = "{}+{}".format(args.tag, subdir)
    save_path = os.path.join(args.base_dir, subdir)
    cmd += " --config.log.save_path " + save_path

    # redirect standard output into logging
    if args.logging is not None:
        cmd += " > {} 2>&1".format(
            os.path.join(save_path, args.logging))
    return cmd, save_path



def run_job(cmd: str, save_path: str) -> int:
    """Run command returned by `generate_command`.

    Args: 
        cmd (str): The command to be executed.
        save_path (str): The directory corresponding to this command.
    Returns:
        exit_code (int): The executed return value of the fermint function.
    """

    # Write this command to `save_path/cmd.sh`.
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cmd_path = os.path.join(save_path, "cmd.sh") 
    with open(cmd_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(cmd)
        f.write("\n")
    
    return os.system(cmd)

def acquire_job(crt_idx, idx_lock, total):
        idx_lock.acquire()
        if crt_idx.value >= total:
            idx_lock.release()
            return -1
        idx = crt_idx.value
        crt_idx.value = crt_idx.value + 1
        idx_lock.release()
        return idx

def worker_func(pid, crt_idx, total, idx_lock, output_lock, args, loop_dict, raw_cmd):
    """Function of each GPU worker. It will keep taking jobs until crt_idx reaches total.

    Args:
        pid (int): The process index of this worker.
        crt_idx (Value): shared variable from multiprocessing. crt_idx.value is the current index.
        total (int): Total number of programs to be executed.
        idx_lock (Lock()): Lock to operate crt_idx
        args (argparse.Namespace): The original input args
        loop_dict (dict): The dictionary of loop variables.
        raw_cmd (str): The raw command written in script.sh.
    """
    while True:
        idx = acquire_job(crt_idx, idx_lock, total)
        if idx == -1:   # If all jobs are done, return
            break
        gpu_list = [args.gpu_list[i] for i in range(pid * args.per_gpu,
                        (pid + 1) * args.per_gpu)]
        
        # Print info about this job
        output_lock.acquire()
        print("########    {}\n    Worker {} obtains job id {} and starts in gpu {}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), pid, idx, gpu_list
        ))
        cmd, save_path = generate_command(args, idx, loop_dict, raw_cmd, gpu_list)
        print("    The command is\n    {}\n########\n".format(cmd))
        output_lock.release()

        # Start running quitely
        exit_code = run_job(cmd, save_path)

        # Print complete info about this job
        output_lock.acquire()
        print("########    {}\n    Worker {} finishes job id {}. The return value is {}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), pid, idx, exit_code))
        print("########\n")
        output_lock.release()


if __name__ == '__main__':
    args, loop_key = getArgs()
    time.sleep(0.5)

    # Read and prepare looping for the command.
    print("-------- command --------")
    with open(os.path.join(args.base_dir, args.script), "r") as f:
        while True:
            raw_cmd = f.readline()
            if raw_cmd[:6] != "#!/bin":
                break
    if raw_cmd[-1] == "\n":
        raw_cmd = raw_cmd[:-1]

    print("Your basic command read from {} is \n{}".format(
        os.path.join(args.base_dir, args.script), raw_cmd
    ))
    # Construct the loop key dict, and check the raw command is as desire.
    loop_dict = {
        key[5:]: vars(args)[key] for key in loop_key
    }
    
    for key in loop_dict.keys():
        assert "--config."+key not in raw_cmd, "The changable args `{}` has already been specified in the script, which is not allowed.".format(key)
    assert "--config.log.save_path" not in raw_cmd, "The script command has specified `save_path`. However, this should be generated by the program automatically."
    
    total_runs = np.prod([len(values) for values in loop_dict.values()])
    num_worker = len(args.gpu_list) // args.per_gpu
    print("{} programs will be executed. {} process will be created to run them parallelly.".format(total_runs, num_worker))
    print("-------- command --------\n\n")

    # Prepare multi processing shared variables    
    current_idx = Value('i', 0)
    idx_lock, output_lock = Lock(), Lock()
    jobs = []

    # Create processes
    for worker in range(num_worker):
        p = Process(target = worker_func, args = (worker, current_idx, total_runs, idx_lock, output_lock, args, loop_dict, raw_cmd, ))
        jobs.append(p)
        p.start()

    # Wait and join them    
    for job in jobs:
        job.join()

    print("Finshed sweep function.")




    
