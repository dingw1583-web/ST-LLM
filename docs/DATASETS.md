# Dataset Preparation

Class-directory names are case-sensitive. Create every configured class directory and place one bidirectional TCP flow per `.pcap`. The scripts read these directories directly; categorization templates do not move captures automatically.

Dataset sources: [ISCX-VPN2016](https://www.unb.ca/cic/datasets/vpn.html), [ISCX-Tor2016](https://www.unb.ca/cic/datasets/tor.html). Obtain USTC-TFC2016 from its dataset provider.

## ISCX-VPN

CLI name: `iscx-vpn`. Classes: 6.

| Label | Class directory |
| ---: | --- |
| 0 | `data/ISCX-VPN/Chat/` |
| 1 | `data/ISCX-VPN/Email/` |
| 2 | `data/ISCX-VPN/File/` |
| 3 | `data/ISCX-VPN/P2P/` |
| 4 | `data/ISCX-VPN/Streaming/` |
| 5 | `data/ISCX-VPN/VoIP/` |

## ISCX-NonVPN

CLI name: `iscx-nonvpn`. Classes: 6.

| Label | Class directory |
| ---: | --- |
| 0 | `data/ISCX-NonVPN/Chat/` |
| 1 | `data/ISCX-NonVPN/Email/` |
| 2 | `data/ISCX-NonVPN/File/` |
| 3 | `data/ISCX-NonVPN/Streaming/` |
| 4 | `data/ISCX-NonVPN/Video/` |
| 5 | `data/ISCX-NonVPN/VoIP/` |

## ISCX-Tor

CLI name: `iscx-tor`. Classes: 8.

| Label | Class directory |
| ---: | --- |
| 0 | `data/ISCX-Tor/Audio-Streaming/` |
| 1 | `data/ISCX-Tor/Browsing/` |
| 2 | `data/ISCX-Tor/Chat/` |
| 3 | `data/ISCX-Tor/File/` |
| 4 | `data/ISCX-Tor/Mail/` |
| 5 | `data/ISCX-Tor/P2P/` |
| 6 | `data/ISCX-Tor/Video-Streaming/` |
| 7 | `data/ISCX-Tor/VoIP/` |

## ISCX-NonTor

CLI name: `iscx-nontor`. Classes: 8.

| Label | Class directory |
| ---: | --- |
| 0 | `data/ISCX-NonTor/Audio/` |
| 1 | `data/ISCX-NonTor/Browsing/` |
| 2 | `data/ISCX-NonTor/Chat/` |
| 3 | `data/ISCX-NonTor/Email/` |
| 4 | `data/ISCX-NonTor/FTP/` |
| 5 | `data/ISCX-NonTor/P2P/` |
| 6 | `data/ISCX-NonTor/Video/` |
| 7 | `data/ISCX-NonTor/VoIP/` |

## USTC-TFC2016

CLI name: `ustc-tfc2016`. Classes: 20.

| Label | Class directory |
| ---: | --- |
| 0 | `data/USTC-TFC2016/Benign_BitTorrent/` |
| 1 | `data/USTC-TFC2016/Benign_Facetime/` |
| 2 | `data/USTC-TFC2016/Benign_FTP/` |
| 3 | `data/USTC-TFC2016/Benign_Gmail/` |
| 4 | `data/USTC-TFC2016/Benign_MySQL/` |
| 5 | `data/USTC-TFC2016/Benign_Outlook/` |
| 6 | `data/USTC-TFC2016/Benign_Skype/` |
| 7 | `data/USTC-TFC2016/Benign_SMB/` |
| 8 | `data/USTC-TFC2016/Benign_Weibo/` |
| 9 | `data/USTC-TFC2016/Benign_WorldOfWarcraft/` |
| 10 | `data/USTC-TFC2016/Malware_Cridex/` |
| 11 | `data/USTC-TFC2016/Malware_Geodo/` |
| 12 | `data/USTC-TFC2016/Malware_Htbot/` |
| 13 | `data/USTC-TFC2016/Malware_Miuref/` |
| 14 | `data/USTC-TFC2016/Malware_Neris/` |
| 15 | `data/USTC-TFC2016/Malware_Nsis-ay/` |
| 16 | `data/USTC-TFC2016/Malware_Shifu/` |
| 17 | `data/USTC-TFC2016/Malware_Tinba/` |
| 18 | `data/USTC-TFC2016/Malware_Virut/` |
| 19 | `data/USTC-TFC2016/Malware_Zeus/` |

`DATA/` contains empty categorization templates for each configuration. Lowercase `data/` is the default location for local captures and generated arrays/graphs. Override it with `ST_LLM_DATA_ROOT`. Original capture manifests, packet captures, processed data, and trained weights are not distributed.
