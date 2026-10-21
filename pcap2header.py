import argparse
import os
import numpy as np
from scapy.all import *
from config import *

HEADER_FIXED_LENGTH = 50
PAD_BYTE = 0


def process_header(raw_header):

    header_bytes = np.frombuffer(raw_header, dtype=np.uint8)

    if len(header_bytes) > HEADER_FIXED_LENGTH:
        processed = header_bytes[:HEADER_FIXED_LENGTH]
    else:
        processed = np.pad(
            header_bytes,
            (0, HEADER_FIXED_LENGTH - len(header_bytes)),
            mode="constant",
            constant_values=PAD_BYTE
        )
    return processed


def process_pcap_file(file_path):

    p_header_list = []

    def _process(pkt):
        if pkt.haslayer("Raw"):
            raw_packet = bytes(pkt)

            raw_header = raw_packet[:-len(pkt["Raw"].load)] if pkt["Raw"] else raw_packet
        else:
            raw_header = bytes(pkt)
        processed_header = process_header(raw_header)
        p_header_list.append(processed_header)

    sniff(offline=file_path, prn=_process, store=0)
    return np.array(p_header_list, dtype=np.uint8)


def pcap2header4ISCX(dir_path_dict, save_path_dict):

    for category in dir_path_dict:
        dir_path = dir_path_dict[category]
        file_list = os.listdir(dir_path)
        for file in file_list:
            if not file.endswith('.pcap'):
                continue
            file_path = os.path.join(dir_path, file)
            print(f'Processing {file_path}...')
            headers = process_pcap_file(file_path)


            save_file = file[:-5] + '_header.npz'
            np.savez_compressed(
                os.path.join(save_path_dict[category], save_file),
                header=headers
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['iscx-vpn', 'iscx-nonvpn', 'iscx-tor', 'iscx-nontor', 'ustc-tfc2016'])
    opt = parser.parse_args()


    config = get_config(opt.dataset)


    pcap2header4ISCX(
        dir_path_dict=config.DIR_PATH_DICT, save_path_dict=config.DIR_PATH_DICT
    )
