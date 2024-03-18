from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib.packet import in_proto
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib import hub
from ryu import cfg
from rf import MachineLearningAlgo
from gb import GBClassification

# 0 - Data collection, 1 - Detection
APP_MODE = 1

# 0 - Normal Traffic, 1 - Attack Traffic.
TRAFFIC_TYPE = 1

COLLECTOR_INTERVAL = 10
DETECTION_INTERVAL = 10

class DDOSMLApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def _init_(self, *args, **kwargs):
        super(DDOSMLApp, self)._init_(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.rfobj = None
        self.gbobj = None  
        if APP_MODE == 1:
            self.rfobj = MachineLearningAlgo()  
            self.gbobj = GBClassification("dataset.csv") # Instantiate GBClassification
        self.monitor_thread = hub.spawn(self.monitor)

    def monitor(self):
        self.logger.info("start flow monitoring thread")
        interval = DETECTION_INTERVAL
        while True:
            hub.sleep(interval)
            for datapath in self.datapaths.values():
                ofp = datapath.ofproto
                ofp_parser = datapath.ofproto_parser
                req = ofp_parser.OFPFlowStatsRequest(datapath)
                datapath.send_msg(req)

    @set_ev_cls([ofp_event.EventOFPFlowStatsReply, ], MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        for stat in body:
            if (stat.priority != 1):
                continue
            if int(stat.match['ip_proto']) == 1:
                self.ddos_detection(ev.msg.datapath.id, stat.match['eth_src'], stat.duration_sec, stat.match['ip_proto'], 0,
                                    0, stat.byte_count, stat.packet_count)
            elif int(stat.match['ip_proto']) == 6:
                self.ddos_detection(ev.msg.datapath.id, stat.match['eth_src'], stat.duration_sec, stat.match['ip_proto'],
                                    stat.match['tcp_src'], stat.match['tcp_dst'], stat.byte_count, stat.packet_count)
            elif int(stat.match['ip_proto']) == 17:
                self.ddos_detection(ev.msg.datapath.id, stat.match['eth_src'], stat.duration_sec, stat.match['ip_proto'],
                                    stat.match['udp_src'], stat.match['udp_dst'], stat.byte_count, stat.packet_count)

    def ddos_detection(self, datapath_id, source_mac, duration, ip_proto, src_port, dst_port, byte_count, packet_count):
        print("Extract the flow params ", duration, ip_proto, src_port, dst_port, byte_count, packet_count)
        dpid = datapath_id
        src = source_mac

        if APP_MODE == 1:
            if self.rfobj is not None:
                ids = self.rfobj.classify([[duration, ip_proto, src_port, dst_port, byte_count, packet_count]])
                print("Detection Result:", ids)
                if ids == 1 and self.gbobj is not None:
                    # If an attack is detected, perform classification
                    attack_type = self.gbobj.classify([[duration, ip_proto, src_port, dst_port, byte_count, packet_count]])
                   
                    self.logger.info("DDoS Detected (%s) from %s ...blocking it", attack_type, src)
                    datapath = self.datapaths[dpid]
                    ofproto = datapath.ofproto
                    parser = datapath.ofproto_parser
                    match = parser.OFPMatch(eth_src=src)
                    action = []
                    self.add_flow(datapath, 100, match, action, idle_t=120)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.datapaths[datapath.id] = datapath
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, idle_t=0, hard_t=0 )


    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_t=60, hard_t=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    idle_timeout=idle_t, hard_timeout=hard_t,                                    
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    idle_timeout=idle_t, hard_timeout=hard_t,                
                                    match=match, instructions=inst)

        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]
        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:

            # check IP Protocol and create a match for IP
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                srcip = ip.src
                dstip = ip.dst
                protocol = ip.proto

                # if ICMP Protocol
                if protocol == in_proto.IPPROTO_ICMP:
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, in_port=in_port, eth_src=src, ipv4_src=srcip,
                                            eth_dst=dst, ipv4_dst=dstip, ip_proto=protocol)

                #  if TCP Protocol
                elif protocol == in_proto.IPPROTO_TCP:
                    t = pkt.get_protocol(tcp.tcp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, in_port=in_port, eth_src=src, ipv4_src=srcip,
                                            eth_dst=dst, ipv4_dst=dstip, ip_proto=protocol, tcp_src=t.src_port,
                                            tcp_dst=t.dst_port, )

                #  If UDP Protocol
                elif protocol == in_proto.IPPROTO_UDP:
                    u = pkt.get_protocol(udp.udp)
                    match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, in_port=in_port, eth_src=src, ipv4_src=srcip,
                                            eth_dst=dst, ipv4_dst=dstip, ip_proto=protocol, udp_src=u.src_port,
                                            udp_dst=u.dst_port, )

                # flow_mod & packet_out
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_t=10)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions, idle_t=10)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
