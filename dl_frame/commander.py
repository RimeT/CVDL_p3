import os
from configparser import ConfigParser


def _get_config_handler(config_path):
    cp = ConfigParser()
    cp.read(config_path)
    return cp


def training_command(config_path):
    arguments = list()
    arguments.append('python')
    arguments.append('tools/mxnet/training.py')
    ch = _get_config_handler(config_path)
    for node in ch:
        for item in ch[node]:
            value = ch[node][item]
            if len(value) > 0:
                arguments.append("--{}={}".format(item, value))
    return " ".join(arguments)


def testing_command(config_path):
    arguments = list()
    arguments.append('python')
    arguments.append('tools/mxnet/testing.py')
    ch = _get_config_handler(config_path)
    for item in ch['job']:
        arguments.append("--{}={}".format(item, ch['job'][item]))
    return " ".join(arguments)
    # return 'pwd'


def execute(stage, config_path):
    if stage == 'train':
        command_str = training_command(config_path)
    elif stage == 'test':
        command_str = testing_command(config_path)
    os.system(command_str)


if __name__ == "__main__":
    # execute('train', '../config/mx_det_config.ini')
    execute('test', '../config/mx_test_det_config.ini')
