%matplotlib ipympl
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from resutils import graph2json as g2json
from mpl_toolkits.mplot3d import Axes3D
from resutils import percolator
from resutils import netfitter2 as netfitter
from resutils import graph2json as g2json
from resutils import utilities
from resutils import plott
import networkx as nx
from tqdm import tqdm_notebook, tnrange
import time
import pandas as pd
from collections import OrderedDict
import json
import scipy
from scipy.signal import chirp, spectrogram
import copy
import requests
import gzip
import base64
import time
import igraph as ig
import random
from scipy import signal
# from tqdm import tqdm

perc = percolator.Percolator(serverUrl=" http://landau-nic0.mse.ncsu.edu:15143/percolator/")
nf_cpu = netfitter.NetworkFitter(serverUrl="http://landau-nic0.mse.ncsu.edu:15123/symphony/")
nf_lancuda = netfitter.NetworkFitter(serverUrl="http://landau-nic0.mse.ncsu.edu:15123/symphony/")


# run this cell

def create_cylinder_network(key=None, N=200, L=100, lengthDev=0, D=0.1, diamDev=0, proximity=0.1, angleDev=90, seed=0,
                            box=100, boy=100, boz=100, air=False, is3D=True, tag='string', ellipsoidal=False):
    datac = perc.get_default_config()
    datac['cylinder']['enabled'] = True
    datac['cylinder']['angleX'] = 0
    datac['cylinder']['angleZ'] = 0
    datac['cylinder']['angleDev'] = angleDev
    datac['cylinder']['diamDev'] = diamDev
    datac['cylinder']['number'] = N
    datac['cylinder']['diameter'] = D
    datac['cylinder']['length'] = L
    datac['cylinder']['lengthDev'] = lengthDev
    datac['simulation']['boxDimensionX'] = box
    datac['simulation']['boxDimensionY'] = boy
    datac['simulation']['boxDimensionZ'] = boz
    datac['simulation']['proximity'] = proximity
    datac['simulation']['seed'] = seed
    datac['simulation']['withAir'] = air
    datac['simulation']['is3D'] = is3D
    datac['simulation']['tag'] = tag
    datac['simulation']['steps'] = 0
    datac['simulation']['isEllipsoidal'] = ellipsoidal

    #     key1=perc.create()
    #     print(key1)
    if key != None:
        network = perc.generate_net(key, **datac)
    else:
        print("No key provided")
    #     perc.clear(key1)
    # curset = perc.export_network(key1)
    # print(curset)
    #     perc.delete(key1)

    return datac


def create_sphere_network():
    return 0


# def create_spaghetti_network(N=200,L=100,nsegments=10,lengthDev=0,D=0.1,diamDev=0,proximity=0.1,angleDev=10,seed=0,box=100,boy=100,boz=100,tag='Ag', is3D=False,):

#     datac=perc.get_default_config()
#     datac['spaghetti']['enabled']=True
#     datac['spaghetti']['angleX']=0
#     datac['spaghetti']['angleZ']=0
#     datac['spaghetti']['angleDev']=angleDev
#     datac['spaghetti']['firstAngleDev']=180
#     datac['spaghetti']['diamDev']=diamDev
#     datac['spaghetti']['number']=N
#     datac['spaghetti']['diameter']=D
#     datac['spaghetti']['length']=L
#     datac['spaghetti']['numberOfSegments']=nsegments
#     datac['spaghetti']['lengthDev']=lengthDev

#     #For now, stats module runs statistics only for cylinders, not spaghetti, but we still need to define cylinder just for the purpose of bypassing division by zero error
#     #Here we just need to copy and paste parameters from spaghetti network and disable cylinders. This will be fixed in future versions.
#     datac['cylinder']['enabled']=False
#     datac['cylinder']['angleX']=0
#     datac['cylinder']['angleZ']=0
#     datac['cylinder']['angleDev']=angleDev
#     datac['cylinder']['diamDev']=diamDev
#     datac['cylinder']['number']=N
#     datac['cylinder']['diameter']=D
#     datac['cylinder']['length']=L
#     datac['cylinder']['lengthDev']=lengthDev

#     datac['simulation']['boxDimensionX'] = box
#     datac['simulation']['boxDimensionY'] = boy
#     datac['simulation']['boxDimensionZ'] = boz
#     datac['simulation']['proximity'] = proximity
#     datac['simulation']['seed'] = seed
#     datac['simulation']['withAir'] = False
#     datac['simulation']['is3D'] = is3D
#     datac['simulation']['steps'] = 0
#     datac['simulation']['tag'] = tag


#     key1=perc.create()
# #     print(key1)

#     network = perc.generate_net(key1,**datac)
# #     network = perc.generate_net(key1,**datac)
# #     perc.clear(key1)
#     # curset = perc.export_network(key1)
#     # print(curset)
#     perc.clear(key1)
#     perc.delete(key1)

#     return network

def create_spaghetti_network(key=None, N=200, L=100, angleX=0, angleY=0, nsegments=10, lengthDev=0, D=0.1,
                             diamDev=0, proximity=0.1, primaryAngleDev=10, angleDev=10, seed=0, box=100, boy=100,
                             boz=100, air=False, is3D=True, tag='Ag'):
    datac = perc.get_default_config()
    datac['spaghetti']['enabled'] = True
    datac['spaghetti']['angleX'] = angleX
    datac['spaghetti']['angleZ'] = angleY
    datac['spaghetti']['angleDev'] = angleDev
    datac['spaghetti']['firstAngleDev'] = primaryAngleDev
    datac['spaghetti']['diamDev'] = diamDev
    datac['spaghetti']['number'] = N
    datac['spaghetti']['diameter'] = D
    datac['spaghetti']['length'] = L
    datac['spaghetti']['numberOfSegments'] = nsegments
    datac['spaghetti']['lengthDev'] = lengthDev

    # For now, stats module runs statistics only for cylinders, not spaghetti, but we still need to define cylinder just for the purpose of bypassing division by zero error
    # Here we just need to copy and paste parameters from spaghetti network and disable cylinders. This will be fixed in future versions.
    datac['cylinder']['enabled'] = False
    datac['cylinder']['angleX'] = 0
    datac['cylinder']['angleZ'] = 0
    datac['cylinder']['angleDev'] = angleDev
    datac['cylinder']['diamDev'] = diamDev
    datac['cylinder']['number'] = N
    datac['cylinder']['diameter'] = D
    datac['cylinder']['length'] = L
    datac['cylinder']['lengthDev'] = lengthDev

    datac['simulation']['boxDimensionX'] = box
    datac['simulation']['boxDimensionY'] = boy
    datac['simulation']['boxDimensionZ'] = boz
    datac['simulation']['proximity'] = proximity
    datac['simulation']['seed'] = seed
    datac['simulation']['withAir'] = air
    datac['simulation']['is3D'] = is3D
    datac['simulation']['steps'] = 0
    datac['simulation']['tag'] = tag

    if key != None:
        #         network = perc.generate_net(key,**datac)
        perc.generate_network(key, **datac)
    else:
        print("No key provided")

    return datac


def create_electrodes(el1mat=[1, 1], el2mat=[1, 1], delta=10):
    els1 = perc.get_electrodes_rects(el1mat, gap=0.3)
    els2 = perc.get_electrodes_rects(el2mat, gap=0.3)
    return els1, els2, delta


# def is_conducting(network,els1,els2,delta,box,boy,boz):
#     try:
#         G = perc.load_graph_from_json(network)
#         xmin, xmax, ymin, ymax, zmin, zmax = perc.get_3d_minmax(G)

#         accepted_graphs = perc.get_connected_graphs_electrodes(supergraph=G,delta=delta,elects1=els1,elects2=els2,box=box,boy=boy,boz=boz)
#     except:
#         return 0

#     if len(accepted_graphs)>0:
#         el1_nodes = perc.get_nodes_for_electrode(elects=els1, pos=perc.get_pos_for_subgraph(accepted_graphs[0],
#                                                                                       nx.get_node_attributes(G, 'pos3d')),
#                                                 xmax=xmin + delta, xmin=xmin - delta, ymax=ymax, zmax=zmax)
#         el2_nodes = perc.get_nodes_for_electrode(elects=els2, pos=perc.get_pos_for_subgraph(accepted_graphs[0],
#                                                                                   nx.get_node_attributes(G, 'pos3d')),
#                                             xmax=xmax + delta, xmin=xmax - delta, ymax=ymax, zmax=zmax)


# #         print(el1_nodes)
# #         print(el2_nodes)
#         return 1.
#     else:
# #         print("No connection!")
#         return 0.

def is_conducting(network, input_electrode, output_electrode):
    G = perc.load_graph_from_json(network)

    accepted_graphs = perc.get_graphs_connecting_electrodearrays(G, {0: input_electrode, 1: output_electrode})
    if len(accepted_graphs) > 0:
        return 1.
    else:
        return 0.


def get_el1_el2_nodes(network, els1, els2, delta, box, boy, boz):
    try:
        G = perc.load_graph_from_json(network)
        xmin, xmax, ymin, ymax, zmin, zmax = perc.get_3d_minmax(G)

        accepted_graphs = perc.get_connected_graphs_electrodes(supergraph=G, delta=delta, elects1=els1, elects2=els2,
                                                               box=box, boy=boy, boz=boz)
    except:
        return 0

    if len(accepted_graphs) > 0:
        el1_nodes = perc.get_nodes_for_electrode(elects=els1, pos=perc.get_pos_for_subgraph(accepted_graphs[0],
                                                                                            nx.get_node_attributes(G,
                                                                                                                   'pos3d')),
                                                 xmax=xmin + delta, xmin=xmin - delta, ymax=ymax, zmax=zmax)
        el2_nodes = perc.get_nodes_for_electrode(elects=els2, pos=perc.get_pos_for_subgraph(accepted_graphs[0],
                                                                                            nx.get_node_attributes(G,
                                                                                                                   'pos3d')),
                                                 xmax=xmax + delta, xmin=xmax - delta, ymax=ymax, zmax=zmax)

        #         print(el1_nodes)
        #         print(el2_nodes)
        return el1_nodes, el2_nodes
    else:
        #         print("No connection!")
        return 0.


def plot_network(network, els1, els2, delta):
    # plot_nxgraph(accepted_graphs[1],nx.get_node_attributes(accepted_graphs[1],'pos'))
    G = perc.load_graph_from_json(network)
    graph = G
    title_str = "Edges: {}, TPVF: {:1.3f}, L/R: {}".format(len(graph.edges), network['stat']['TPVF'],
                                                           network['stat']['aspect'])
    # network['stat']['aspect']=cylL*2/cylD
    # network['stat']['boxVol']=boxX*boxY*boxZ
    # network['stat']['cylN']=cylN
    ax = perc.plot_pos3d(graph, title=title_str)
    xmin, xmax, ymin, ymax, zmin, zmax = perc.get_3d_minmax(graph)
    perc.plot_electrodes(xmax=xmin, ymax=ymax, zmax=zmax, ax=ax, els=els1, xdelta=delta)
    perc.plot_electrodes(xmax=xmax, ymax=ymax, zmax=zmax, ax=ax, els=els2, xdelta=delta)


def get_ids_for_elemtype(elements, elemtype='MemristorElm'):
    elems = json.loads(elements)['elements']
    ids = []
    for elem in elems:
        if elemtype.lower() in elem['type'].lower():
            ids.append(elem['elementId'])
    return ids


def get_tracked_edgeid_kv(graph, circuit, elements, measurables_ids):
    ggg = g2json.get_currents_for_graph(graph, circuit, elements)
    return {k: v for k, v in nx.get_edge_attributes(ggg, 'elementid').items() if v in measurables_ids}


def is_conducting(network, input_electrode, output_electrode):
    G = perc.load_graph_from_json(network)

    accepted_graphs = perc.get_graphs_connecting_electrodearrays(G, {0: input_electrode, 1: output_electrode})
    if len(accepted_graphs) > 0:
        return 1.
    else:
        return 0.

box, boy, boz = 250, 250, 250
#G = perc.load_graph_from_json(network)
#to open what you saved
import dill as pickle
with open('ag2s_2492_mem.pkl', 'rb') as file:
    G = pickle.load(file)
input_electrode_arr=perc.create_elect_boxes(elmat=[1,1],plane=0,gap=0.0,box={'x0':0,'y0':0,'z0':0,'x1':box,'y1':boy,'z1':boz},delta=(12.5,0,0))
output_electrode_arr=perc.create_elect_boxes(elmat=[1,1],plane=1,gap=0.0,box={'x0':0,'y0':0,'z0':0,'x1':box,'y1':boy,'z1':boz},delta=(12.5,0,0))
#delta is half of electrode thickness
accepted_graphs=perc.get_graphs_connecting_electrodearrays(G,{0:input_electrode_arr,1:output_electrode_arr})

accepted_graphs=perc.get_graphs_connecting_electrodearrays(G,{0:input_electrode_arr,1:output_electrode_arr})
try:
    accepted_graph=accepted_graphs[0]
except:
    accepted_graph=G

fig = perc.plotly_pos3d(graph=accepted_graph,is3d=False,plot_wires=True,rescolor='rgba(220, 20, 20, 0.3)')
fig=perc.plotly_electrode_boxes(fig=fig,el_array=input_electrode_arr)
fig=perc.plotly_electrode_boxes(fig=fig,el_array=output_electrode_arr)

#fig.show()

input_nodes=perc.get_nodes_for_box_array(pos=nx.get_node_attributes(accepted_graph,'pos3d'),el_array=input_electrode_arr)
output_nodes=perc.get_nodes_for_box_array(pos=nx.get_node_attributes(accepted_graph,'pos3d'),el_array=output_electrode_arr)

wired_electrodes_graph=perc.wire_nodes_on_electrodes(accepted_graph,[input_nodes,output_nodes])

el_pan = []
for el_arr in [input_nodes, output_nodes]:
    sub_el = []
    for elk in el_arr.keys():
        sub_el.append(el_arr[elk][0])
    el_pan.append(sub_el)

comb_graph = wired_electrodes_graph
comb_graph = perc.prune_dead_edges_el_pan(wired_electrodes_graph, el_pan, runs=32)
comb_graph = perc.precondition_trim_lu(comb_graph,el_pan,cutoff=1e-3)

comb_graph = perc.convert_edgeclass_to_device(comb_graph,mem='Zn',res='Ag',diode='sph')
print(len(comb_graph.edges()))

fig = perc.plotly_pos3d(graph=comb_graph,is3d=True,plot_wires=True,rescolor='rgba(220, 20, 20, 0.3)')
fig=perc.plotly_electrode_boxes(fig=fig,el_array=input_electrode_arr)
fig=perc.plotly_electrode_boxes(fig=fig,el_array=output_electrode_arr)

fig.update_layout(title="Wires with up to 0° curviness")
fig.show()

#triangle pulses, pos-pos-neg-neg

volt = 5
samp = 1000
total_time = 10
time = total_time / samp
#peaks = 5 #freq
#t = np.linspace(0, peaks*2, samp)
#y = signal.sawtooth(2 * np.pi * 1 * t+pi/2,0.5)

peaks = 20 #freq
t = np.linspace(0, samp*time, samp)
y = signal.sawtooth(np.pi * (peaks/(samp*time)) * t+np.pi/2,0.5)

#off by 1 error with zero index???

neg_ind1 = int(math.ceil(samp/2))
neg_ind2 = int(samp)

y = abs(y)
y[neg_ind1:neg_ind2] = y[neg_ind1:neg_ind2] * -1
y = y*volt

plt.figure()
plt.plot(t,y)
plt.xlabel("Time[s]")
plt.ylabel("Amplitude[V]")
plt.title("Input Voltage")
plt.show()

from matplotlib.colors import Normalize
import matplotlib.cm as cmx
import matplotlib.cm as cm
from collections import OrderedDict

#circ = g2json.transform_network_to_circuit_res_cutoff(graph=comb_graph, inels=ins, outels=outs, Ron_pnm=1, Roff_pnm=1000, mobility=2.56e-9, nw_res_per_nm=0.01, t_step="1e-5", scale=1e-6,mem_cutoff_len_nm=0)

def transform_network_to_circuit_res_cutoff(graph, inels=[], outels=[], contels=[], mobility=2.56E-9, Ron_pnm=100,
                                            Roff_pnm=1000, nw_res_per_nm=0.005, junct_res_per_nm=500, t_step="5e-6",
                                            scale=1e-9, elemceil=10000, randomized_mem_width=False,
                                            mem_cutoff_len_nm=10):
    pos3d = nx.get_node_attributes(graph, 'pos3d')
    #import pdb; pdb.set_trace()
    #     el_type='m'
    rndmzd = randomized_mem_width
    # memristor base configuration

    #     Ron = 500.
    #     Roff = 100000.
    #     totwidth = 1.0E-8
    #     dopwidth = 0.5*totwidth

    add_junct_res_to_wire = 'air' not in nx.get_edge_attributes(graph,'edgeclass').values()

    drainres = 100

    elemceil = elemceil  # maximum id of element

    edges = graph.edges()
    elemtypes = nx.get_edge_attributes(graph, 'edgetype')
    elemclasses = nx.get_edge_attributes(graph, 'edgeclass')
    doc = {}
    doc[0] = ['$', 1, t_step, 10.634267539816555, 43, 2.0, 50]

    mnresistances = []
    mxresistances = []
    mrresistances = []
    for elemid, e in enumerate(edges, 1):

        # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
        p1 = np.array(pos3d[e[0]])
        p2 = np.array(pos3d[e[1]])
        length = np.linalg.norm(p1 - p2) * scale * 1e9
        # totwidth = length * 1e-9
        # dopwidth = length * 0.5 * 1e-9
        totwidth = length * 1e-9
        dopwidth = length * 0.5 * 1e-9
        Ron = Ron_pnm * length
        Roff = Roff_pnm * length
        try:
            el_type = elemtypes[e]
            el_class = elemclasses[e]
        except:
            el_type = 'w'
            print("Can't obtain edge properties")
            pass
        if el_type == 'm' and length > mem_cutoff_len_nm:
            totwidth_rnd = totwidth  # + random.uniform(-totwidth / 5., totwidth / 5.)
            dopwidth_rnd = random.uniform(0., totwidth_rnd)
            mxresistances.append(Roff)
            mnresistances.append(Ron)
            if (p1[0] - p2[0] > 0) and (np.random.rand() > 1):
                lst = ["m", e[1], e[0], 0, elemid, str(Ron), str(Roff), str(dopwidth_rnd if rndmzd else dopwidth),
                       str(totwidth_rnd if rndmzd else totwidth), str(mobility)]
            else:
                lst = ["m", e[0], e[1], 0, elemid, str(Ron), str(Roff), str(dopwidth_rnd if rndmzd else dopwidth),
                       str(totwidth_rnd if rndmzd else totwidth), str(mobility)]
        elif el_type == 'r' or (el_type == 'm' and length <= mem_cutoff_len_nm):
            if 'air' in el_class:
                lst = ['r', e[0], e[1], 0, elemid, str(length * junct_res_per_nm)]
            else:
                mrresistances.append(length * nw_res_per_nm)
                lst = ['r', e[0], e[1], 0, elemid, str(length * nw_res_per_nm + junct_res_per_nm if add_junct_res_to_wire else 0)]
        elif el_type == 'd':
            lst = ["d", e[0], e[1], 1, elemid, "0.805904"]
        elif el_type == 'w':
            lst = ["w", e[0], e[1], 1, elemid]
        doc[elemid] = lst

    # nodes = list(G.nodes)

    #     inoutnodes = random.sample(nodes, nin + nout)

    inputids = []
    outputids = []
    controlids = []

    for node in inels:
        elemid += 1
        elemceil -= 1
        # lst = ["R", k, elemceil, 0, elemid, "2", "40.0", "0.0", "0.0", "0.0", "0.5"]
        #         idk=random.choice(inels[k])
        idk = node
        lst = ["R", idk, elemceil, 0, elemid, "0", "40.0", "0.0", "0.0", "0.0", "0.5"]
        doc[elemid] = lst
        inputids.append(elemid)

    for node in contels:
        elemid += 1
        elemceil -= 1
        # lst = ["R", k, elemceil, 0, elemid, "2", "40.0", "0.0", "0.0", "0.0", "0.5"]1
        #         idk=random.choice(inels[k])
        idk = node
        lst = ["R", idk, elemceil, 0, elemid, "0", "40.0", "0.0", "0.0", "0.0", "0.5"]
        doc[elemid] = lst
        controlids.append(elemid)

    for node in outels:
        elemid += 1
        elemceil -= 1
        #         idk=random.choice(outels[k])
        idk = node
        lst = ["r", idk, elemceil, 0, elemid, str(drainres)]
        doc[elemid] = lst
        outputids.append(elemid)

        elemid += 1
        elemsav = elemceil
        elemceil -= 1
        lst = ["g", elemsav, elemceil, 0, 0]
        doc[elemid] = lst

    result = {}
    result['circuit'] = json.dumps(doc)
    result['inputids'] = [f for f in inputids]
    result['outputids'] = [f for f in outputids]
    result['controlids'] = [f for f in controlids]

    return result

import plotly.graph_objects as go


def plot_meas(meas, value='current'):
    t = []
    m_table = {}
    if type(meas) == list:
        new_meas = {}
        new_meas['measurements'] = meas
        meas = new_meas

    for step in meas['measurements']:
        t.append(step['time'])
        for record_key in step['records'].keys():
            if record_key not in m_table.keys():
                m_table[record_key] = []
            m_table[record_key].append(step['records'][record_key][value])

    plt.figure()

    for measurable in m_table.keys():
        plt.plot(t, m_table[measurable], label=str(measurable))

    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.legend()
    plt.show()
    return m_table


def get_t_mtable(meas, value='current'):
    t = []
    m_table = {}
    if type(meas) == list:
        new_meas = {}
        new_meas['measurements'] = meas
        meas = new_meas

    for step in meas['measurements']:
        t.append(step['time'])
        for record_key in step['records'].keys():
            if record_key not in m_table.keys():
                m_table[record_key] = []
            m_table[record_key].append(step['records'][record_key][value])

    return t, m_table


def plotly_meas(meas, value='current', title='', ivt_scale=1):
    assert ivt_scale > 0., "ivt_scale should be > 0"

    t, m_table = get_t_mtable(meas, value)
    #     plt.figure()
    fig = go.Figure()
    for measurable in m_table.keys():
        fig.add_trace(
            go.Scatter(x=(np.array(t) * ivt_scale).tolist(), y=(np.array(m_table[measurable]) / ivt_scale).tolist(),
                       mode='lines',
                       name=str(measurable)
                       ))
    #         plt.plot(t,m_table[measurable],label=str(measurable))
    fig.update_layout(title=title,
                      xaxis_title='Time (s)',
                      yaxis_title='Current (A)')

    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Current (A)')
    #     plt.legend()
    #     plt.show()
    return fig, m_table

def get_currents_for_graph(graph,circuit,currents):

    elcur={}
    for element in json.loads(currents)['elements']:
        elcur[element['elementId']]=element['current']

    dutcirc=json.loads(circuit['circuit'])

    for k in list(dutcirc.keys())[1:]:
        dutcirc[k]
        ed=(dutcirc[k][1],dutcirc[k][2])
        eid=dutcirc[k][4]
        if dutcirc[k][0] not in ['R','g']:
            try:
                cur=elcur[eid]
                graph.edges()[ed]['elementid']=eid
                graph.edges()[ed]['current']=cur
            except:
#                 print("Auxilary element {} can't be used in the circuit".format(dutcirc[k]))
                pass

    return graph


from matplotlib.colors import Normalize
import matplotlib.cm as cmx
import matplotlib.cm as cm
from collections import OrderedDict


def plot_pos3d_lightning(graph=None, ax=None, title='', is3d=True, plot_wires=True, save_as=None, elev=20, azim=90,
                         max_current=1, cmap='jet', dist=5, max_line_width=5, min_line_width=0.4, electrodes=[]):
    pos3d = nx.get_node_attributes(graph, 'pos3d')
    #     max_current=np.max(np.abs(list(nx.get_edge_attributes(graph,'current').values())))
    #     max_line_width = 5
    #     min_line_width = 0.4
    jet = cm = plt.get_cmap(cmap)
    cNorm = Normalize(vmin=-max_current, vmax=max_current)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    if ax == None:
        fig = plt.figure(figsize=(10, 10))
        #         ax = fig.gca(projection="3d")
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    #         ax.scatter(xs, ys, zs, c='r', s=5)
    # ax.plot(xs,ys,zs, color='r')
    for e in graph.edges():
        x1 = pos3d[e[0]][0]
        y1 = pos3d[e[0]][1]
        z1 = pos3d[e[0]][2] if is3d else 0.
        x2 = pos3d[e[1]][0]
        y2 = pos3d[e[1]][1]
        z2 = pos3d[e[1]][2] if is3d else 0.
        x = [x1, x2]
        y = [y1, y2]
        z = [z1, z2]
        edgetype = {}

        try:
            edgetype = graph[e[0]][e[1]]['edgetype']
            edgecurrent = graph[e[0]][e[1]]['current']
            #             colorVal = scalarMap.to_rgba(abs(edgecurrent))

            do_plot = True

            if ('w' in edgetype):
                if not plot_wires:
                    do_plot = False
                    pass

            if do_plot:
                lw = abs(edgecurrent / max_current) * max_line_width
                lw = min_line_width if lw < min_line_width else lw

                if 'm' in edgetype:
                    p = ax.plot(x, y, z, color='g', linewidth=lw)
                #                     p = ax.scatter(x1, y1, z1, c='r',s=lw)
                else:
                    p = ax.plot(x, y, z, color='b', linewidth=lw)

        #                 if 'm' in edgetype:
        #                     p = ax.plot(x, y, z, color=scalarMap.to_rgba(abs(edgecurrent)), linewidth=lw)
        #                 else:
        #                     p = ax.plot(x, y, z, color=scalarMap.to_rgba(abs(0)), linewidth=lw)

        #             if 'm' in edgetype:
        #                 ax.plot(x, y, z, c='b', label='memristor')
        #             elif 'r' in edgetype:
        #                 ax.plot(x, y, z, c='m', label='resistor')
        #             elif 'w' in edgetype:
        #                 if plot_wires:
        #                     ax.plot(x, y, z, c='g', label='wire')
        #             elif 'd' in edgetype:
        #                 ax.plot(x, y, z, c='orange', label='diode')
        except:
            ax.plot(x, y, z, c='k')
            pass

    # Remove background axis color
    ax.set_facecolor((0, 0, 0, 0))
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Bonus: To get rid of the grid as well:
    ax.grid(False)

    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.dist = dist
    #     sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=-0.001, vmax=0.001))
    #     plt.colorbar(sm)
    for el, col in zip(electrodes, ['y', 'b', 'g', 'r', 'c', 'm'] * 50):
        plot_electrode_boxes(ax=ax, el_array=el, cols=[col])

    if save_as == None:
        plt.show()
    else:
        fig.savefig(save_as)
        plt.close()
    return ax


def plot_electrode_boxes(ax=None, el_array=None, cols=['k', 'g', 'b', 'r', 'c', 'm', 'y', 'k']):
    #     x1, x2 = xmax - xdelta, xmax + xdelta
    colors = cols * 10
    if ax == None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    for k, n in zip(list(el_array.keys()), range(len(list(el_array.keys())))):
        x1, y1, z1, x2, y2, z2 = el_array[k]['x0'], el_array[k]['y0'], el_array[k]['z0'], el_array[k]['x1'], \
                                 el_array[k]['y1'], el_array[k]['z1']
        edges = []
        edges.append([[y1, z1, x1], [y1, z2, x1]])
        edges.append([[y1, z2, x1], [y2, z2, x1]])
        edges.append([[y2, z2, x1], [y2, z1, x1]])
        edges.append([[y2, z1, x1], [y1, z1, x1]])

        edges.append([[y1, z1, x2], [y1, z2, x2]])
        edges.append([[y1, z2, x2], [y2, z2, x2]])
        edges.append([[y2, z2, x2], [y2, z1, x2]])
        edges.append([[y2, z1, x2], [y1, z1, x2]])

        edges.append([[y1, z1, x1], [y1, z1, x2]])
        edges.append([[y1, z2, x1], [y1, z2, x2]])
        edges.append([[y2, z2, x1], [y2, z2, x2]])
        edges.append([[y2, z1, x1], [y2, z1, x2]])

        for e in edges:
            y = np.array(e)[:, 0].tolist()
            z = np.array(e)[:, 1].tolist()
            x = np.array(e)[:, 2].tolist()
            ax.plot(x, y, z, colors[n])
    return ax


# # nx.get_edge_attributes(current_graphs[2],'edgetype')
# mc=[]
# for cg in current_graphs:
#     mc.append(np.max(np.abs(list(nx.get_edge_attributes(cg,'current').values()))))
# np.max(mc)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm_notebook


def generate_weighted_ws_graph(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    for (u, v) in G.edges():
        G.edges[u, v]['current'] = random.random()
    return G


def graph_entropy(graph):  # use this one
    currents = np.abs(list(nx.get_edge_attributes(graph, 'current').values()))
    total = np.sum(currents)
    if total == 0:
        return 0
    entropy = []
    for cur in currents:
        a = cur / total
        entropy.append(-a * np.log(a))
    entropy = np.array(entropy)
    entropy[np.isnan(entropy)] = 0
    return np.sum(entropy)


def get_norm_laplacian(graph, weight=None):
    L = nx.laplacian_matrix(graph, weight=weight).todense()
    L = L / scipy.trace(L)
    L = np.abs(L)
    return L


def vn_entropy_graph(graph, weight=None):
    L = get_norm_laplacian(graph, weight=weight)
    L = scipy.absolute(L)
    return vn_entropy_laplacian(L)


def vn_entropy_laplacian(L):
    eigvals = np.real(scipy.linalg.eigvals(L))
    eigvals = np.abs(eigvals)
    eigvals = eigvals[np.nonzero(eigvals)]
    eigvals[np.isnan(eigvals)] = 0
    return -np.sum(eigvals * np.log(eigvals))


def kl_divergence(a, b):
    aa = get_norm_laplacian(a)
    bb = get_norm_laplacian(b)
    return np.trace(np.matmul(aa, np.log(aa) - np.log(bb)))


def js_divergence(a, b):
    aa = get_norm_laplacian(a)
    bb = get_norm_laplacian(b)
    uu = 0.5 * (aa + bb)
    dist = vn_entropy_laplacian(uu) - 0.5 * (vn_entropy_laplacian(aa) + vn_entropy_laplacian(bb))
    return dist


def get_laplacian(graph, weight=None):
    L = nx.laplacian_matrix(graph, weight=weight)
    return L


# def [time eig_max VNGE]=VNGE_FINGER(A)

def get_weighted_adjacency_mat(G, weight=None):
    A = nx.to_scipy_sparse_matrix(G, weight=weight, format='csr')
    return A


def get_FINGER_VNGE(g, weight=None):
    # A=L
    A = get_weighted_adjacency_mat(g, weight)
    A = scipy.absolute(A)  # we want all currents to be positive
    d = scipy.sum(A, 1);
    if scipy.sum(d) == 0.:
        return 0.
    c = 1 / scipy.sum(d);
    n = A.shape[0];
    edge_weight = scipy.nonzero(A);

    edge_weight = A[edge_weight]
    GEapprox = 1 - c ** 2 * (scipy.sum(scipy.square(d)) + scipy.sum(scipy.square(edge_weight)));

    D = scipy.sparse.diags(np.array(d).reshape(-1, ));
    L = D - A;
    eig_max = scipy.real(c * scipy.sparse.linalg.eigs(L, 1)[0][0]);
    VNGE = -GEapprox * np.log(eig_max);
    return VNGE

# d=sum(A,2); c=1/sum(d);  n=size(A,1);
# edge_weight=nonzeros(A);
# GEapprox=1-c**2*(sum(d.^2)+sum(edge_weight**2));

# D=spdiags(d,0,n,n); L=D-A;
# eig_max=c*eigs(L,1);
# VNGE=-GEapprox*log(eig_max);
# # #     return VNGE

#how to set the parameters for the strukov model

ins = el_pan[0][:]
outs = el_pan[1][:]
mems = sum([v == 'm' for v in nx.get_edge_attributes(comb_graph, 'edgetype').values()])
print("Total mems: ", mems)
ress = sum([v == 'r' for v in nx.get_edge_attributes(comb_graph, 'edgetype').values()])
print("Total res: ", ress)
wires = sum([v == 'w' for v in nx.get_edge_attributes(comb_graph, 'edgetype').values()])
print("Total wires: ", wires)
# circ = g2json.transform_network_to_circuit(graph=comb_graph, inels=ins, outels=outs, Ron_pnm=1, Roff_pnm=1000, mobility=2.56e-9, nw_res_per_nm=0.002, t_step="5e-6", scale=1e-6)
circ = transform_network_to_circuit_res_cutoff(graph=comb_graph, inels=ins, outels=outs, Ron_pnm=2, Roff_pnm=2000, mobility=7e-7, nw_res_per_nm=0.01, t_step="1e-5", scale=1e-6,mem_cutoff_len_nm=0)
circ = g2json.modify_integration_time(circ, set_val='1e-4')

# get the output currents by setting the voltage, waiting for eq_time, then calcualting the response, then repeat for next voltage input
# Esc->M->Enter to turn on markup mode, add #'s for smaller text'
# remove brackets from y[n] if defining y with np.full


utils = utilities.Utilities(serverUrl=nf_lancuda.serverUrl)
key = nf_lancuda.init_steps(circ['circuit'], utils)
res = {}
current_graphs = []
current_values = []
voltage_values = []
entropy_list = []

for n in tqdm_notebook(range(len(y))):  # range here should be same as # samples
    res[n] = nf_lancuda.make_step(key, X=[y[n]], inputids=circ['inputids'], outputids=circ['outputids'], controlids=[],
                                  eq_time=time, utils=utils)
    currents = utils.getElementsIVs(key)
    gg = g2json.get_currents_for_graph(comb_graph, circ, currents).copy()  # change graph to comb_graph
    current_graphs.append(gg)
    entropy_list.append(graph_entropy(gg))
    current_dict = nx.get_edge_attributes(gg, 'current')
    current_list = list(current_dict.values())
    current_values.append(current_list)

    voltage_dict = nx.get_edge_attributes(gg, 'voltage')
    voltage_list = list(voltage_dict.values())
    voltage_values.append(voltage_list)

    # voltages and currents for all elements?? need total yeah?

nf_lancuda.complete_steps(key, utils)
res = {0: res}

#     currents=utils.getElementsIVs(key)
#     gg=g2json.get_currents_for_graph(comb_graph,circ,currents).copy()
#     current_graphs.append(gg)

# print(utils.statistics(key))
# nf_lancuda.complete_steps(key,utils)
# res={0:res}

xs=g2json.batch_plot_single_sim(res,title='Output Current',tstep=time) #what you want to run
plt.tight_layout()

def one_list(old_list):
    new_list = []
    for sublist in old_list:
        for item in sublist:
            new_list.append(item)
    return new_list
currents = one_list(xs)

plt.style.use('classic')

fig,ax = plt.subplots()
fig.patch.set_facecolor('w')
ax.margins(0.05)

plt.plot(y, currents,'-',color="tab:blue")
#plt.plot(voltage,currents)
plt.ylabel("Current (A)")
plt.xlabel("Voltage (V)")
plt.show()
plt.tight_layout()

#change name of csv file!

df = pd.DataFrame({'voltage':y[:-1],'currents':currents[1:],'time':t[1:],'Entropies':entropy_list[1:]})
df.to_csv('big_net_final_entropy_1.csv')

max(currents)

#test a frame
#replace max current with max current found in cell above

#%%time
#540 and 660
plt.style.use('dark_background')
# ax=None
ax=plot_pos3d_lightning(graph=current_graphs[660],is3d=False,elev=-90,azim=-90,plot_wires=False,max_current=1.603e-7,dist=5,min_line_width=.6,max_line_width=25,electrodes=[input_electrode_arr,output_electrode_arr])
# plot_electrode_boxes(ax=ax,el_array=input_electrode_arr,cols=['g'])
# plot_electrode_boxes(ax=ax,el_array=output_electrode_arr,cols=['w'])

#CHANGE FILE FOR IMAGES EVERY TIME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##need to create folder for images before running this

import cv2

%matplotlib inline
from multiprocessing.pool import Pool
from itertools import repeat

fnames=[]
for i in tqdm_notebook(range(len(current_graphs[:]))):
    fnames.append("/home/jovyan/work/LabMeasurements/Damien/Big_net_images_2/img_{}.png".format(i)) #need to create folder for images before running this

#(graph=None, ax=None, title='', is3d=True, plot_wires=True, save_as=None, elev=20, azim=90, max_current=1,cmap='jet',dist=5,max_line_width=5,min_line_width=0.4,electrodes=[]):

#ax=plot_pos3d_lightning(graph=current_graphs[660],is3d=False,elev=-90,azim=-90,plot_wires=False,max_current=6.332e-8,dist=5,min_line_width=.6,max_line_width=25,electrodes=[input_electrode_arr,output_elec

with Pool(processes=20) as pool:
    results = pool.starmap(plot_pos3d_lightning,zip(current_graphs[:],repeat(None),repeat(''),repeat(False),repeat(False),fnames,repeat(-90),repeat(-90),repeat(1.603e-7),repeat('jet'),repeat(5),repeat(25),repeat(0.6),repeat([input_electrode_arr,output_electrode_arr])))
print("done")

#make video
#use folder with images

import os
import subprocess
image_output_dir="/home/jovyan/work/LabMeasurements/Damien/Big_net_images_1"
framerate=15
curdir=os.getcwd()
os.chdir(image_output_dir)
subprocess.call('echo yes | ffmpeg -framerate {} -i "img_%d.png" output_15.mp4'.format(framerate),shell=True)
os.chdir(curdir)


