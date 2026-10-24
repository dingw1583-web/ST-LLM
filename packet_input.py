from pathlib import Path
import numpy as np


def read_packets(path):
    if Path(path).suffix.lower() == '.npz':
        with np.load(path,allow_pickle=True) as f:
            result={key:f[key].copy() for key in f.files}
        result['header_masked']=bool(result.get('header_masked',False))
        return result
    if Path(path).suffix.lower() not in ('.pcap','.pcapng'):
        raise ValueError('Expected per-flow NPZ, PCAP or PCAPNG: '+str(path))
    from scapy.all import PcapReader, IP, IPv6, TCP, UDP, raw
    headers,payloads,times=[],[],[]
    with PcapReader(str(path)) as reader:
        for packet in reader:
            if not (IP in packet or IPv6 in packet):
                continue
            transport=packet.getlayer(TCP) if TCP in packet else packet.getlayer(UDP)
            if transport is None:
                continue
            network=packet.getlayer(IP) if IP in packet else packet.getlayer(IPv6)
            network_bytes=raw(network)
            if IP in packet:
                prefix=network_bytes[:12]
            else:
                prefix=network_bytes[:8]
            data=raw(transport.payload)
            transport_bytes=raw(transport)
            transport_header=transport_bytes[:-len(data)] if data else transport_bytes
            headers.append(list(prefix+transport_header[4:]))
            payloads.append(list(data)); times.append(float(packet.time))
    return dict(header=headers,payload=payloads,time=np.asarray(times),header_masked=True)
