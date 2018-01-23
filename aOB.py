# coding: utf-8

import ads as ads
import numpy as np
import pdb
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import csv

import os
curr = os.getcwd()

# token should be stored locally, per ads pkg docs

def find_all_ORCID_papers_in_ADS(verbose=False):
    """
    Extract all papers that have an ORCID ID specified in any of the three ORCID fields (orcid_pub, orcid_user,
    orcid_other). Combines the lists, following the priority order, and outputs to a file.

    :return: Numpy save file ('all_astro_orcid_papers.npy') to a local directory, that contains a list of bibcodes
        and the combined ORCID ID arrays
    """

    start = 0
    allpapers_lol = list([])
    papers = np.arange(2000)
    query = 0
    while len(papers) == 2000:
        req = ads.SearchQuery(q='orcid:000* database:astronomy', rows=2000, start=start,
                                      fl=['bibcode', 'orcid_pub', 'orcid_user', 'orcid_other', 'author'])
        req.execute()
        papers = req.response.docs

        start += 2000
        query += 1
        allpapers_lol.append(papers)

    allpapers = [item for sublist in allpapers_lol for item in sublist]
    if verbose:
        print(len(allpapers))

    j = 0
    for paper in allpapers:
        try:
            authorid = []
            orcpub = (paper['orcid_pub'] != None)
            orcuser = (paper['orcid_user'] != None)
            orcother = (paper['orcid_other'] != None)
            if orcpub and orcuser and orcother:
                if len(paper['orcid_pub']) != len(paper['orcid_user']) or len(paper['orcid_pub']) != len(paper['orcid_other']):
                    paper['authorid'] = paper['orcid_pub']
                    continue
            for i in range(len(paper['orcid_pub'])):
                if orcpub and paper['orcid_pub'][i] != '-':
                    authorid.append(paper['orcid_pub'][i])
                elif orcuser and paper['orcid_user'][i] != '-':
                    authorid.append(paper['orcid_user'][i])
                elif orcother and paper['orcid_other'][i] != '-':
                    authorid.append(paper['orcid_other'][i])
                else:
                    authorid.append('-')
            paper['authorid'] = authorid
            j += 1
        except TypeError:
            paper['authorid'] = paper['orcid_pub']
        except IndexError:
            paper['authorid'] = paper['orcid_pub']
        if j%1000 == 0 and verbose:
            print(j)

    all_astro_orcid_papers = np.asarray(allpapers)
    np.save('all_astro_orcid_papers.npy', all_astro_orcid_papers)

def build_ORCID_network(path=curr,verbose=False):
    """
    Takes output file from find_all_ORCID_papers_in_ADS and converts it to a network, using the ORCID IDs
    as nodes and the bibcodes as edges.

    :param path: path to where output file from find_all_ORCID_papers_in_ADS is stored
    :return: None; outputs graph to file
    """
    orcid_data = np.load(path + '/' + 'all_astro_orcid_papers.npy')

    # get the nodes (unique ORCID IDs)
    bfl = []
    for paper in orcid_data:
        bfl.append(paper['authorid'][:])
    allorcids = [item for sublist in bfl for item in sublist]

    alluorcids = (np.unique(allorcids))[1:]
    if verbose:
        print(len(alluorcids))

    # this is if we want a network with lonely nodes
    #G = nx.Graph()
    #for orcid in alluorcids:
    #    G.add_node(orcid)

    G = nx.Graph()
    for paper in orcid_data:
        good_list = []
        for e in paper['authorid']:
            if e != '-':
                good_list.append(e)
        if len(good_list) > 1:
            # This is if we want a network with no lonely nodes
            for author1 in good_list:
                G.add_node(author1)
            for i, author1 in enumerate(good_list):
                # print(good_list)
                for author2 in good_list[i + 1:]:
                    if author2 not in G[author1]:
                        G.add_edge(author1, author2)

    with open('ORCID_graph.pkl', 'wb') as f:
        pickle.dump(G, f)

    # format for working with Gephi visualization
    nx.write_gexf(G, 'ORCID_graph.gexf')

def calc_centrality(path=curr,verbose=False):
    """
    Calculates the centrality of every node in the graph; returns a sorted list as a file.

    This is the answer to "who is astronomy's Kevin Bacon?"

    :param path: path to where the output graph file is stored
    :return: None; writes file of nodes and centrality scores, sorted by centrality
    """
    with open(path + '/' + 'ORCID_graph.pkl', 'rb') as f:
        G = pickle.load(f)

    clcent = nx.closeness_centrality(G)

    sorti = np.argsort(list(clcent.values()))[::-1]
    sort_orcids = np.array(list(clcent.keys()))[sorti]
    sort_central = np.array(list(clcent.values()))[sorti]

    if verbose:
        print(sort_orcids)
        print(sort_central)

    with open('centrality.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(sort_orcids,sort_central))

def calc_path_2_ORCIDs(path=curr,node1=None,node2=None):
    """
    Calculates shortest path between two nodes (ORCID IDs). Returns path + degrees.

    Answers "How many degrees are you from astronomy's Kevin Bacon (or whoever else)?"
    :param path: path to where the output graph file is stored
    :param node1: first node; defaults to most central node
    :param node2: second node; defaults to second most central node
    :return: shortest path + degrees (len(shortest path) - 1)
    """

    with open(path + '/' + 'ORCID_graph.pkl', 'rb') as f:
        G = pickle.load(f)

    if (node1 is None) or (node2 is None):
        with open(path + '/' + 'centrality.csv', 'rb') as f:
            centrality = csv.reader(f, delimiter='\t')
            rn = 0
            for row in centrality:
                if rn == 0:
                    tmp1 = row
                    rn += 1
                elif rn == 1:
                    tmp2 = row
                    rn += 1
                else:
                    break
        if node1 is None:
            node1 = tmp1[0]
        if node2 is None:
            node2 = tmp2[0]

    try:
        short_path = nx.algorithms.shortest_paths.generic.shortest_path(G, source=node1,target=node2)
    except:
        print('These two ORCID IDs are not connected.')
        return

    print('The shortest path is: ' + ', '.join(short_path))
    print('The two ORCID IDs are connected by {} degree(s).'.format(len(short_path)-1))

def find_coauthors_without_ORCID(path=curr,node=None):
    """
    Given an input ORCID ID, find all of that author's coauthors who have not entered an ORCID ID.

    :param path: path to where the output file from find_all_ORCID_papers_in_ADS is stored
    :param node: ORCID ID to find coauthors of
    :return: list of coauthors who have not entered an ORCID ID
    """

    if node is None:
        print('Please enter an ORCID ID.')
        return

    orcid_data = np.load(path + '/' + 'all_astro_orcid_papers.npy')

    # search data for papers authored by the given ORCID ID and find their coauthors who did not give an ORCID ID
    coauthors = set()
    for i in range(len(orcid_data)):
        if node in orcid_data[i]['authorid']:
            for j, ind in enumerate(orcid_data[i]['authorid']):
                if ind == '-':
                    coauthors.add(orcid_data[i]['author'][j])

    print('Coauthors missing ORCID IDs: ')
    print(sorted(coauthors))
