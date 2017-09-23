import networkx as nx, math, sys, time
verbosed_flag = False

class AnytimeSCANGraph(nx.Graph):
    '''Specific Graph Class for Anytime SCAN algorithm
    attributes:
        untouched_nodes: set, initial value includes all nodes
        unprocessed_nodes: set, includes all nodes from which it has not expanded, initial value is empty set
        processed_nodes: set, includes all nodes from which it has expanded, initial value is empty set

        core_nodes: set, it includes all (processed | unprocessed) core nodes, initial value is empty set. it's used
                    for obtaining clustering results easiy
        core_subgraph: nx.Graph, is the subgraph obtained by core_nodes and it only includes the edges with
                        SS >= epsilon between core nodes
        cluster_num: int, current cluster numbers
        degree_map: dict, key is the degree and value is the set of nodes with corresponding degree value
                    degree here does not count the node itself. This dict is used from active learning,
                    it only include the unprocessed nodes
    '''


    def __init__(self, filename = None):
        '''
        initialize the undirected unweighted graph from a edge list file
        :param filename: edge list file, each line represents an edge, with format "node1 node2"
        '''
        super(AnytimeSCANGraph, self).__init__()
        if filename == None: return

        fgraph = open(filename)
        for line in fgraph:
            tokens = line.strip().split()
            self.add_edge(tokens[0], tokens[1])
            self.edge[tokens[0]][tokens[1]]['SS'] = -1
        print('num of nodes: %d; num of edges: %d.' % (self.number_of_nodes(), self.number_of_edges()))

        self.untouched_nodes = set()
        self.unprocessed_nodes = set()
        self.processed_nodes = set()
        self.core_nodes = set()
        '''cluster_num is not used in this version'''
        self.cluster_num = 0
        '''degree_map is used for active learning, which only record the deree distribution of unprocessed nodes'''
        self.degree_map = dict()
        self.core_subgraph = None
        '''low and high are used for active learning'''
        self.low = 9999999
        self.high = 0
        '''candidate_nodes_AL is core candiate for active learing'''
        self.candidate_nodes_AL = None
        '''candidate_nodes_AL_non_core is non core candiate for active learing'''
        self.candidate_nodes_AL_non_core = None

        '''initialize the status of nodes'''
        for node in self.nodes():
            self.node[node]['category'] = 'noise'
            self.node[node]['clusterID'] = -1
            self.node[node]['numStrongEdges'] = 0
            self.untouched_nodes.add(node)
            self.degree_map.setdefault(self.degree(node), set()).add(node)

    def anytime_scan(self, dataset_name, epsilon, mu, threshold = 100, isHeuristic = True, isRandom_AL = False, true_label_filename = None):
        '''
        Perform anytime SCAN,
        :param dataset_name:
        :param epsilon and mu: two parameters in SCAN
        :param threshold: stop condition for active anytime SCAN. It can be the num of active learning picks or running time.
                            Now we only implements the version of num of picks by active learning
        :param isHeuristic: if apply heuristic in initial clustering. The default value is True, i.e. apply heuristic in initia clustering
        :param isRandom_AL: if apply random strategy or high-low strategy in active learning. The default value is False, i.e. apply high-low stratedy in active learning
        :param true_label_filename: the file name store the true label of each node.
                                The default value is None, and does not calculate the ARI and NMI;
                                Otherwise, calculatet the ARI and NMI
        :return: None
        '''

        print('Entering anytime_scan()...')
        print('dataset: %s; isHeuristic: %s; isRandom_AL: %s; epsilon: %f; mu: %d; theshold: %d.' % (dataset_name, isHeuristic, isRandom_AL, epsilon, mu, threshold))
        filename = '%s_results_%s_%s_%.2f_%d_%d.txt' % (dataset_name, isHeuristic, isRandom_AL, epsilon, mu, threshold)
        fout = open(filename, 'w')
        line = 'Iter\tARI\tNMI\tModu\tnumSS\tnumCore\ttime\ttotal_time\tlow\thigh\n'
        fout.write(line)

        true_cluster_dict = None
        if true_label_filename != None:
            true_cluster_dict = self.get_true_cluster(true_label_filename)

        start = time.clock()

        if isHeuristic:
            self.get_initial_clustering_heuristic(epsilon, mu)
        else:
            self.get_initial_clustering(epsilon, mu)

        end = time.clock()

        total_time  = float(end- start)

        print('current clusers after initial:')
        start = time.clock()
        initial_clusters, num_SS, num_Core = self.output_current_clustering_result(epsilon)
        end = time.clock()

        print('time for initial clustering: %f.' % (total_time + end - start, ))

        ARI = -1
        NMI = -1
        if true_label_filename != None:
            ARI, NMI = self.calculate_metrics_current(true_cluster_dict, initial_clusters)
            print('ARI: %f; NMI: %f.' % (ARI, NMI))
        modularity = self.calculate_modularity_current(initial_clusters)
        print('current modularity: %f.' % modularity)

        line = '%d\t%.4f\t%.4f\t%.4f\t%d\t%d\t%.4f\t%.4f\t%d\t%d\n' % (0, ARI, NMI, modularity, num_SS, num_Core, total_time, (total_time + end - start), self.low, self.high)
        fout.write(line)

        counter = 0
        while counter < threshold and len(self.unprocessed_nodes) > 0:
            '''Pick an unprocessed node by active learning and expand from it'''
            '''Perform the active learning based on the selected strategy'''
            start = time.clock()

            if not isRandom_AL:
                next_seed = self.active_learning(mu)

            if next_seed == None:
                print('No seed can be selected in active learning. done!!!')
                break

            print('picked seed:',next_seed)

            if not isRandom_AL:
                self.expand_onehop_from_node(next_seed, epsilon, mu)

            '''move the seed from unprocessed to processed'''
            '''active_learning_random() will remove seed from unprocessed, so we need check if unprocessed node have the seed before remove it'''
            if next_seed in self.unprocessed_nodes:
                self.unprocessed_nodes.remove(next_seed)
            self.processed_nodes.add(next_seed)

            counter += 1

            end = time.clock()
            total_time += (end - start)


            print('in active learning %d' % (counter,))

            if counter % 1000 == 0:
                print('current clusers:')
                start = time.clock()
                clusters, num_SS, num_Core = self.output_current_clustering_result(epsilon)
                end = time.clock()

                print('time neded: %f.' % (total_time + end - start,))

                if true_label_filename != None:
                    ARI, NMI = self.calculate_metrics_current(true_cluster_dict, clusters)
                    print('ARI: %f; NMI: %f.' % (ARI, NMI))
                modularity = self.calculate_modularity_current(clusters)
                print('current modularity: %f.' % modularity)

                line = '%d\t%.4f\t%.4f\t%.4f\t%d\t%d\t%.4f\t%.4f\t%d\t%d\n' % (
                counter, ARI, NMI, modularity, num_SS, num_Core, total_time, (total_time + end - start), self.low,
                self.high)
                fout.write(line)

                if int(NMI) == 1:
                    break


        print('Anytime SCAN done!!!')
        print('Num of active learning: %d' % (counter,))
        print('final clusers:')
        start = time.clock()
        clusters, num_SS, num_Core = self.output_final_clustering_result(epsilon)
        end = time.clock()
        print('time neded: %f.' % (total_time + end - start,))
        '''
        total_num = 0
        for key in clusters:
            print(str(key), ':', sorted(clusters[key]))
            total_num += len(clusters[key])
        print('total_num in clustering results: %d' % (total_num,))
        '''

        if true_label_filename != None:
            ARI, NMI = self.calculate_metrics_final(true_cluster_dict)
            print('ARI: %f; NMI: %f.' % (ARI, NMI))
        modularity = self.calculate_modularity_final()
        print('final modularity: %f.' % modularity)
        print('finihed!')

        line = '%d\t%.4f\t%.4f\t%.4f\t%d\t%d\t%.4f\t%.4f\t%d\t%d\n' % (counter, ARI, NMI, modularity, num_SS, num_Core, total_time, (total_time + end - start), self.low, self.high)
        fout.write(line)
        fout.close()
        '''output the core / noise'''
        filename = 'node_info_%s_results_%s_%s_%.2f_%d_%d.txt' % (dataset_name, isHeuristic, isRandom_AL, epsilon, mu, threshold)
        self.print_nodes(filename)


    def original_scan(self, dataset_name, epsilon, mu, true_label_filename = None):
        '''

        :param dataset_name: dataset name, used for name result file
        :param epsilon:
        :param mu:
        :param true_label_filename: ground truth label fille, used for coputing ARI and NMI, if applicable
        :return:
        '''
        print('Entering original_scan()...')
        print('dataset: %s; epsilon: %.4f; mu: %d.' % (dataset_name,  epsilon, mu))

        '''unprocessed nodes: nodes which have no expanded from
            processed nodes: nodes which have expanded from
        '''
        self.unprocessed_nodes |= self.untouched_nodes
        self.untouched_nodes.clear()
        print('num unprocessed: %d; num processed: %d.' % (len(self.unprocessed_nodes), len(self.processed_nodes)))

        self.cluster_num = 0
        num_of_cores = 0
        seed_pool = set()

        start = time.clock()

        while(len(self.unprocessed_nodes) > 0):
            seed = self.unprocessed_nodes.pop()
            seed_pool.add(seed)
            '''a flag indicate if a new cluster is constructed'''
            has_new_cluster = False
            while(len(seed_pool) > 0):
                expand_seed = seed_pool.pop()
                self.processed_nodes.add(expand_seed)
                if expand_seed in self.unprocessed_nodes:
                    self.unprocessed_nodes.remove(expand_seed)

                if verbosed_flag:
                    print('seed: '+ expand_seed)

                adj_list = nx.all_neighbors(self, expand_seed)

                '''compute / obtain the edges' SS for enighbors'''
                for one_neighbor in adj_list:

                    '''
                    if self.edge[expand_seed][one_neighbor]['SS'] == -1:
                        ss = self.calculate_SS_scan(expand_seed, one_neighbor)
                        self.edge[expand_seed][one_neighbor]['SS'] = ss
                    else:
                        ss = self.edge[expand_seed][one_neighbor]['SS']
                    '''
                    ss = self.calculate_SS_scan(expand_seed, one_neighbor)
                    self.edge[expand_seed][one_neighbor]['SS'] = ss

                    if ss >= epsilon:
                        self.node[expand_seed]['numStrongEdges'] += 1

                '''expand seed is a core'''
                if self.node[expand_seed]['numStrongEdges'] + 1 >= mu:
                    has_new_cluster = True
                    self.node[expand_seed]['category'] = 'core'
                    self.node[expand_seed]['clusterID'] = self.cluster_num

                    num_of_cores += 1

                    if verbosed_flag:
                        print('new core: ' + expand_seed)

                    '''add neighbors in the seed pool'''
                    adj_list = nx.all_neighbors(self, expand_seed)
                    for one_neighbor in adj_list:
                        '''add all un-expanded neighbors wih SS >= epsilon into the seed pool'''
                        if one_neighbor not in self.processed_nodes and self.edge[expand_seed][one_neighbor]['SS'] >= epsilon:
                            seed_pool.add(one_neighbor)
                            self.node[one_neighbor]['clusterID'] = self.cluster_num
                            if verbosed_flag:
                                print('add neighbor: ' + one_neighbor)

            if has_new_cluster:
                self.cluster_num += 1
            '''end for expanding from one seed'''
        '''end for all expanding'''

        final_clusters = self.get_final_clustering_result_scan()
        print('final clusters:')
        print('num of clusters: %d; num of cores: %d.' % (self.cluster_num, num_of_cores))

        end = time.clock()
        print('time needed: %.4f.' % (end - start, ))

        if true_label_filename != None:
            true_cluster_dict = self.get_true_cluster(true_label_filename)
            ARI, NMI = self.calculate_metrics_current(true_cluster_dict, final_clusters)
            print('ARI: %f; NMI: %f.' % (ARI, NMI))
        modularity = self.calculate_modularity_current(final_clusters)
        print('current modularity: %f.' % modularity)

        '''output the core / noise'''
        filename = 'scan_results_node_info_%s_%.2f_%d.txt' % (dataset_name, epsilon, mu)
        self.print_nodes(filename)


    def expand_onehop_from_node(self, next_seed, epsilon, mu):
        '''
        Expand one hop from next_seed obained by active learning
        :param next_seed:
        :param epsilon:
        :param mu:
        :return: None
        '''
        adj_list = nx.all_neighbors(self, next_seed)
        for one_neighbor in adj_list:
            '''only calculate the edge with neighbors i unprocessed nodes'''
            if one_neighbor in self.unprocessed_nodes:
                ss = self.calculate_SS(next_seed, one_neighbor, epsilon)
                self.edge[next_seed][one_neighbor]['SS'] = ss
                if ss >= epsilon:
                    self.node[next_seed]['numStrongEdges'] += 1
                    self.node[one_neighbor]['numStrongEdges'] += 1
                    if self.node[one_neighbor]['numStrongEdges'] + 1 >= mu and self.node[one_neighbor]['category'] != 'core':
                        self.node[one_neighbor]['category'] = 'core'
                        self.core_nodes.add(one_neighbor)

                        '''check if the new core degree will modify the low and high'''
                        if self.degree(one_neighbor) < self.low:
                            for key in range(self.degree(one_neighbor), self.low):
                                if key in self.degree_map:
                                    self.candidate_nodes_AL |= (self.degree_map[key] & self.core_nodes)
                                    self.candidate_nodes_AL_non_core |= (self.degree_map[key] - self.core_nodes)
                            self.low = self.degree(one_neighbor)

                        if self.degree(one_neighbor) > self.high:
                            for key in range(self.high + 1, self.degree(one_neighbor) + 1):
                                if key in self.degree_map:
                                    self.candidate_nodes_AL |= (self.degree_map[key] & self.core_nodes)
                                    self.candidate_nodes_AL_non_core |= (self.degree_map[key] - self.core_nodes)
                            self.high = self.degree(one_neighbor)

        '''End of for loop'''
        if self.node[next_seed]['numStrongEdges'] + 1 >= mu:
            self.node[next_seed]['category'] = 'core'
            self.core_nodes.add(next_seed)

    def update_core_subgraph(self, new_core, epsilon):
        '''
        Insert the new_core and edges with SS > epsilon into the current core_subgraph
        :param new_core:
        :param epsilon:
        :return:
        '''
        self.core_subgraph.add_node(new_core)
        adj_list = set(nx.all_neighbors(self, new_core))
        common_cores = adj_list.intersection(self.core_nodes)
        if verbosed_flag:
            print('new core: ', new_core)
            print('common cores:', common_cores)
        for core in common_cores:
            if self[new_core][core]['SS'] >= epsilon and core in self.unprocessed_nodes:
                self.core_subgraph.add_edge(new_core, core)
                if verbosed_flag: print('insert edge: %s, %s' % (new_core, core))



    def active_learning(self, mu):
        '''
        In the firs time pick of active learning, construct the set of candidate_nodes_AL_non_core and core candidate_nodes_AL with degree in [low, high]
        First active learning strategy: non-core nodes in the candidate set
        Second active learning strategy: core candidate set

        :return: the picked node
        '''
        if verbosed_flag: print('Enering active_learning')

        '''In the 1st active learning, initialize the self.candidate_nodes_AL and self.candidate_nodes_AL_non_core'''
        if self.candidate_nodes_AL == None:
            for core in self.core_nodes:
                if self.degree(core) < self.low: self.low = self.degree(core)
                if self.degree(core) > self.high: self.high = self.degree(core)

            self.candidate_nodes_AL = set()
            for key in range(self.low, self.high + 1):
                if key in self.degree_map:
                    self.candidate_nodes_AL |= self.degree_map[key]

            self.candidate_nodes_AL_non_core = self.candidate_nodes_AL - self.core_nodes
            self.candidate_nodes_AL -= self.candidate_nodes_AL_non_core

        print('core degree: low: %d; high: %d' % (self.low, self.high))

        if self.low > self.high:
            print('low > high, stop the programe')
            return None


        if len(self.candidate_nodes_AL_non_core) > 0:
            print('perform step 1 in active learning')
            return  self.candidate_nodes_AL_non_core.pop()

        if len( self.candidate_nodes_AL) > 0:
            print('perform step 2 in active learning')
            return  self.candidate_nodes_AL.pop()

        '''need expand the candidate degrees'''
        while len(self.candidate_nodes_AL_non_core) == 0 and len( self.candidate_nodes_AL) == 0:
            if self.low >= mu:
                if (self.low - 1) in self.degree_map:
                    self.candidate_nodes_AL |= (self.degree_map[self.low - 1] & self.core_nodes)
                    self.candidate_nodes_AL_non_core |= (self.degree_map[self.low - 1] - self.core_nodes)
                self.low -= 1

            if (self.high + 1) <= max(self.degree_map):
                if (self.high + 1) in self.degree_map:
                    self.candidate_nodes_AL |= (self.degree_map[self.high + 1] & self.core_nodes)
                    self.candidate_nodes_AL_non_core |= (self.degree_map[self.high + 1] - self.core_nodes)
                self.high += 1
            else:
                break

        if len(self.candidate_nodes_AL_non_core) > 0:
            print('perform step 3 in active learning')
            return  self.candidate_nodes_AL_non_core.pop()

        if len( self.candidate_nodes_AL) > 0:
            print('perform step 4 in active learning')
            return  self.candidate_nodes_AL.pop()

        return None


    def get_initial_clustering_heuristic(self, epsilon, mu):
        '''
        To calcluate the initial clustering in anytime SCAN:
                First, construct a node set with degrees >= (mu - 1)
                Then, repeatedly pick node randomly from untouched candidate nodes, expand one-hop to get initial clustering.

        :param epsilon: SS threshold
        :param mu: minimal number of strong edges
        :return: None
        '''
        if verbosed_flag: print('Enering get_initial_clustering_heuristic')

        sorted_degree_list = sorted(self.degree_map)
        candidate_low = mu - 1
        candidate_high = sorted_degree_list[len(sorted_degree_list) - 1]

        print('candidate_low: %d; candidate_high: %d' % (candidate_low, candidate_high))

        candidate_nodes_initial = set()
        candidate_nodes_bak = set()
        for key in range(candidate_low, candidate_high+1):
            if key in self.degree_map:
                candidate_nodes_initial |= self.degree_map[key]
                candidate_nodes_bak |= self.degree_map[key]
        '''count is for debug use'''
        count =0
        while len(candidate_nodes_initial) > 0:
            expand_seed = candidate_nodes_initial.pop()
            self.untouched_nodes.remove(expand_seed)
            self.expand_onehop_from_node_initial_heuristic(candidate_nodes_initial, expand_seed, epsilon, mu)
            '''move the seed node to processed_nodes and remove it from degree_map'''
            self.processed_nodes.add(expand_seed)
            self.degree_map[self.degree(expand_seed)].remove(expand_seed)
            '''remove the empty degre/nodes pair'''
            if len(self.degree_map[self.degree(expand_seed)]) ==0:
                del self.degree_map[self.degree(expand_seed)]

            if verbosed_flag:
                print('counter:',count,'picked node:',expand_seed)
                self.print_nodes()
                self.print_edges()

            count += 1

        if len(self.untouched_nodes) > 0:
            self.unprocessed_nodes |= self.untouched_nodes
            self.untouched_nodes.clear()
            print('mark here: untoched node set is not empty after initial')

        print('num of picks: %d' % (count,))

        if len(self.core_nodes) > 0:
            print('Have already learned core points in heurisic initial clustering')
            return

        count = 0
        candidate_nodes_bak -= self.processed_nodes
        print('num of candidate_nodes_bak: %d; num of processed: %d.' % (len(candidate_nodes_bak), len(self.processed_nodes)))
        while len(self.core_nodes) == 0:
            if len(candidate_nodes_bak) > 0:
                expand_seed = candidate_nodes_bak.pop()
            else:
                print('no core at all.')
                break

            self.expand_onehop_from_node(expand_seed, epsilon, mu)

            '''move the seed from unprocessed to processed and remove it from degree_map'''
            if expand_seed in self.unprocessed_nodes:
                self.unprocessed_nodes.remove(expand_seed)
            self.processed_nodes.add(expand_seed)
            self.degree_map[self.degree(expand_seed)].remove(expand_seed)
            '''remove the empty degre/nodes pair'''
            if len(self.degree_map[self.degree(expand_seed)]) == 0:
                del self.degree_map[self.degree(expand_seed)]

            count += 1

        print('After iterations. num of candidate_nodes_bak: %d; num of processed: %d.' % (len(candidate_nodes_bak), len(self.processed_nodes)))

        candidate_nodes_bak.clear()
        print('%d iterations in 2nd step in heuristic initial clustering.' % (count, ))


        if verbosed_flag: print('Outing get_initial_clustering_heuristic')

    def get_initial_clustering(self, epsilon, mu):
        '''
        To calcluate the initial clustering in anytime SCAN:
                repeatedly pick node randomly from untouched_nodes, expand one-hop to get initial clustering.

        :param epsilon: SS threshold
        :param mu: minimal number of strong edges
        :return: None
        '''
        if verbosed_flag: print('Entering get_initial_clustering')

        '''count is for debug use'''
        count =0
        while len(self.untouched_nodes) > 0:
            expand_seed = self.untouched_nodes.pop()
            self.expand_onehop_from_node_initial(expand_seed, epsilon, mu)
            '''move the seed node to processed_nodes and remove it from degree_map'''
            self.processed_nodes.add(expand_seed)
            self.degree_map[self.degree(expand_seed)].remove(expand_seed)
            '''remove the empty degre/nodes pair'''
            if len(self.degree_map[self.degree(expand_seed)]) ==0:
                del self.degree_map[self.degree(expand_seed)]

            if verbosed_flag:
                print('counter:',count,'picked node:',expand_seed)

            count += 1
        print('num of picks: %d' % (count,))

        if verbosed_flag: print('Outing get_initial_clustering')

    def expand_onehop_from_node_initial(self, expand_seed, epsilon, mu):
        '''
        This function is only called by get_initial_clustering, since it may update the untouched_nodes.
        The similar function called in active learning is defined in another function.
        if the neighbor is in untouched_node, then move the neighbor from untocuhed from unprocessed and calculate the edge's SS
        if the neighbor is in unproessed_node, it means the neighbor has already been visited by another seed, only calculate the edge's SS

        :param expand_seed:
        :param epsilon:
        :param mu:
        :return: None
        '''
        adj_list = nx.all_neighbors(self, expand_seed)
        for one_neighbor in adj_list:
            if one_neighbor in self.untouched_nodes:
                self.untouched_nodes.remove(one_neighbor)
                self.unprocessed_nodes.add(one_neighbor)
            '''the one_neighbor has already in the unprocessed_nodes'''
            ss = self.calculate_SS(expand_seed, one_neighbor, epsilon)
            self.edge[expand_seed][one_neighbor]['SS'] = ss
            if ss >= epsilon:
                self.node[expand_seed]['numStrongEdges'] += 1
                self.node[one_neighbor]['numStrongEdges'] += 1
                if self.node[one_neighbor]['numStrongEdges'] + 1 >= mu:
                    self.node[one_neighbor]['category'] = 'core'
                    self.core_nodes.add(one_neighbor)
        '''End of for loop'''
        if self.node[expand_seed]['numStrongEdges'] + 1 >= mu:
            self.node[expand_seed]['category'] = 'core'
            self.core_nodes.add(expand_seed)


    def expand_onehop_from_node_initial_heuristic(self, candidate_nodes_initial, expand_seed, epsilon, mu):
        '''
        This function is only called by get_initial_clustering_heuristic, since it may update the untouched_nodes and candidate_nodes_initial.
        The similar function called in active learning is defined in another function.
        if the neighbor is in untouched_node, then move the neighbor from untocuhed from unprocessed and calculate the edge's SS
        if the neighbor is in the candidate_nodes_initial, also remove the neighbor from candidate_nodes_initial since it is not untouched any more
        if the neighbor is in unproessed_node, it means the neighbor has already been visited by another seed, only calculate the edge's SS

        :param candidate_nodes_initial:
        :param expand_seed:
        :param epsilon:
        :param mu:
        :return: None
        '''
        adj_list = nx.all_neighbors(self, expand_seed)
        for one_neighbor in adj_list:
            if one_neighbor in self.untouched_nodes:
                self.untouched_nodes.remove(one_neighbor)
                self.unprocessed_nodes.add(one_neighbor)
            if one_neighbor in candidate_nodes_initial:
                candidate_nodes_initial.remove(one_neighbor)
            '''the one_neighbor has already in the unprocessed_nodes'''
            ss = self.calculate_SS(expand_seed, one_neighbor, epsilon)
            self.edge[expand_seed][one_neighbor]['SS'] = ss
            if ss >= epsilon:
                self.node[expand_seed]['numStrongEdges'] += 1
                self.node[one_neighbor]['numStrongEdges'] += 1
                if self.node[one_neighbor]['numStrongEdges'] + 1 >= mu:
                    self.node[one_neighbor]['category'] = 'core'
                    self.core_nodes.add(one_neighbor)
        '''End of for loop'''
        if self.node[expand_seed]['numStrongEdges'] + 1 >= mu:
            self.node[expand_seed]['category'] = 'core'
            self.core_nodes.add(expand_seed)

    def get_initial_core_subgraph(self, epsilon):
        '''
        After get_initial_clusterng, construct the core node subgraph, which includes only the core nodes and the edges with SS >= epsilon

        :param epsilon:
        :return: None
        '''
        self.core_subgraph = self.subgraph(list(self.core_nodes))
        for edge in self.core_subgraph.edges():
            if self.core_subgraph[edge[0]][edge[1]]['SS'] < epsilon:
                self.core_subgraph.remove_edge(edge[0],edge[1])
                if verbosed_flag: print('remove edge: %s, %s' % (edge[0], edge[1]))

    def get_final_clustering_result_scan(self):
        '''
        get the final clustering results from scan alrithm
        hubs (ID = -1) and outliers (ID = -2)
        :return: dict as the clusters
        '''
        clusters = dict()
        '''set to store the nodes with clusterID'''
        nodes_with_IDs = set()
        for node in self.nodes():
            if self.node[node]['clusterID'] >= 0:
                if self.node[node]['clusterID'] in clusters:
                    one_cluster = clusters.get(self.node[node]['clusterID'])
                else:
                    one_cluster = set()
                one_cluster.add(node)
                clusters[self.node[node]['clusterID']] = one_cluster

                nodes_with_IDs.add(node)
        print('num of nodes with cIDs: %d.' % (len(nodes_with_IDs), ))

        noise_set = set(self.nodes()) - nodes_with_IDs
        hubs = set()
        outliers = set()
        for node in noise_set:
            neighbors = nx.all_neighbors(self, node)
            cID_set = set()
            for one_neighbor in neighbors:
                if self.node[one_neighbor]['clusterID'] >= 0:
                    cID_set.add(self.node[one_neighbor]['clusterID'])
                    '''check if the num of IDs in neighbors >= 2'''
                    if len(cID_set) > 1:
                        break
            if len(cID_set) > 1:
                hubs.add(node)
                self.node[node]['clusterID'] = -1
            else:
                outliers.add(node)
                self.node[node]['clusterID'] = -2
        clusters[-1] = hubs
        clusters[-2] = outliers
        print('num of hubs: %d; num of outlires: %d.' % (len(hubs), len(outliers)))

        return clusters




    def output_current_clustering_result(self, epsilon):
        '''
        Calculate the current clustering results in the graph
        Step 1: (a). Compute the connected components from core_subgraph, assign a clueterID to each connected component.
                each cluster is represented by a clusterID / nodes pair.
                (b). Based on the edges which have been visited to clusterID to the boder of each core
        Step 2: other nodes without cluster ID are divided furture into hubs (ID = -1) and outliers (ID = -2)
        :return: dict as the clusters
        '''
        self.get_initial_core_subgraph(epsilon)

        ccs = nx.connected_components(self.core_subgraph)
        clusterID = 0
        clusters = dict()
        '''set to store the nodes with clusterID'''
        nodes_with_IDs = set()
        '''total is used to count the num of nodes in all clusters, which is used to check if there is any overlapping in the clusters'''
        total = 0
        for one_cc in ccs:
            one_cluster = set()
            for node in one_cc:
                self.node[node]['clusterID'] = clusterID
                one_cluster.add(node)
                neighbors = nx.all_neighbors(self, node)
                for one_neighbor in neighbors:
                    if self.edge[node][one_neighbor]['SS'] >= epsilon:
                        self.node[one_neighbor]['clusterID'] = clusterID
                        one_cluster.add(one_neighbor)
            clusters[clusterID] = one_cluster
            nodes_with_IDs |= one_cluster
            total += len(one_cluster)
            clusterID += 1

        '''the following is processing hubs and outliers'''
        noise_set = set(self.nodes()) - nodes_with_IDs
        hubs = set()
        outliers = set()
        for node in noise_set:
            neighbors = nx.all_neighbors(self, node)
            cID_set = set()
            for one_neighbor in neighbors:
                if self.node[one_neighbor]['clusterID'] >= 0:
                    cID_set.add(self.node[one_neighbor]['clusterID'])
                    '''check if the num of IDs in neighbors >= 2'''
                    if len(cID_set) > 1:
                        break
            if len(cID_set) > 1:
                hubs.add(node)
                self.node[node]['clusterID'] = -1
            else:
                outliers.add(node)
                self.node[node]['clusterID'] = -2
        clusters[-1] = hubs
        total += len(hubs)
        clusters[-2] = outliers
        total += len(outliers)

        '''check if there is overlapping in clusters'''
        if total > self.number_of_nodes():
            print('Have overlappings. num of IDs: %d. num of nodes: %d.' % (total, self.number_of_nodes()))
        else:
            print('No overlappings')

        '''count the number of SSs the have been calculated'''
        edge_SSs = nx.get_edge_attributes(self, 'SS')
        total =0
        for key in edge_SSs:
            if edge_SSs[key] > 0:
                total += 1
        print('Num of SSs have been calculated: %d' % (total,))
        print('Num of cores now: %d' % (len(self.core_nodes),))

        '''recover the clusterID as the initial value'''
        for node in self.nodes():
            self.node[node]['clusterID'] = -1

        return (clusters, total, len(self.core_nodes))

    def output_final_clustering_result(self, epsilon):
        '''
        Calculate the final clustering results in the graph.
        The only differece beween this function and output_current_clustering_result is that:
        In this function, we will go one hop more from the unprocesed core nodes.
        But in the function current_clustering_result, we do not go further from the unprocessed core
        :return: dict as the clusters
        '''
        self.get_initial_core_subgraph(epsilon)

        ccs = nx.connected_components(self.core_subgraph)

        clusterID = 0
        clusters = dict()
        '''set to store the nodes with clusterID'''
        nodes_with_IDs = set()
        '''total is used to count the num of nodes in all clusters, which is used to check if there is any overlapping in the clusters'''
        total = 0

        count = 0
        for one_cc in ccs:
            one_cluster = set()
            count += 1
            for node in one_cc:
                self.node[node]['clusterID'] = clusterID
                one_cluster.add(node)
                neighbors = nx.all_neighbors(self, node)
                for one_neighbor in neighbors:
                    '''if the edge <node, one_neighbor> has no SS, first calculate SS'''
                    if self.edge[node][one_neighbor]['SS'] == -1:
                        self.edge[node][one_neighbor]['SS'] = self.calculate_SS(node, one_neighbor,epsilon)
                    if self.edge[node][one_neighbor]['SS'] >= epsilon:
                        self.node[one_neighbor]['clusterID'] = clusterID
                        one_cluster.add(one_neighbor)
            clusters[clusterID] = one_cluster
            nodes_with_IDs |= one_cluster
            total += len(one_cluster)
            clusterID += 1

        '''the following is processing hubs and outliers'''
        noise_set = set(self.nodes()) - nodes_with_IDs
        hubs = set()
        outliers = set()
        for node in noise_set:
            neighbors = nx.all_neighbors(self, node)
            cID_set = set()
            for one_neighbor in neighbors:
                if self.node[one_neighbor]['clusterID'] >= 0:
                    cID_set.add(self.node[one_neighbor]['clusterID'])
                    '''check if the num of IDs in neighbors >= 2'''
                    if len(cID_set) > 1:
                        break
            if len(cID_set) > 1:
                hubs.add(node)
                self.node[node]['clusterID'] =-1
            else:
                outliers.add(node)
                self.node[node]['clusterID'] = -2
        clusters[-1] = hubs
        total += len(hubs)
        clusters[-2] = outliers
        total += len(outliers)

        '''check if there is overlapping in clusters'''
        if total > self.number_of_nodes():
            print('Have overlappings. num of IDs: %d. num of nodes: %d.' % (total, self.number_of_nodes()))
        else:
            print('No overlappings')

        '''count the number of SSs the have been calculated'''
        edge_SSs = nx.get_edge_attributes(self, 'SS')
        total = 0
        for key in edge_SSs:
            if edge_SSs[key] > 0:
                total += 1
        print('Num of SSs have been calculated: %d' % (total,))
        print('Num of cores now: %d' % (len(self.core_nodes),))

        return (clusters, total, len(self.core_nodes))

    def calculate_SS(self, node1, node2, epsilon):
        '''
        Calculate the cosine SS between node1 and node2
        Use upper bound approach to speed up the calculation
        :param node1:
        :param node2:
        :return: SS of edge (node1, node2)
        '''
        small_degree = self.degree(node1)
        if self.degree(node2) < self.degree(node1):
            small_degree = self.degree(node2)
        denominator = math.sqrt((self.degree(node1) + 1) * (self.degree(node2) + 1))
        ss_upper_bound = (small_degree + 1) / denominator
        if ss_upper_bound < epsilon:
            return 0

        commons = len(list(nx.common_neighbors(self, node1, node2))) + 2.0

        return commons/denominator

    def calculate_SS_scan(self, node1, node2):
        '''
        Calculate the cosine SS between node1 and node2 without using upper bound, which is used for original scan algorithm
        :param node1:
        :param node2:
        :return: SS of edge (node1, node2)
        '''

        denominator = math.sqrt((self.degree(node1) + 1) * (self.degree(node2) + 1))
        commons = len(list(nx.common_neighbors(self, node1, node2))) + 2.0

        return commons/denominator


    def print_nodes(self):
        '''
        Output the information of all nodes
        :return: None
        '''
        print('print nodes information:')
        for node in self.nodes():
            print(node, ':',self.node[node]['category'],self.node[node]['clusterID'],self.node[node]['numStrongEdges'])

        print('untouched:')
        print(self.untouched_nodes)
        print('unprocessed:')
        print(self.unprocessed_nodes)
        print('processed:')
        print(self.processed_nodes)
        print('core:')
        print(self.core_nodes)

    def print_nodes(self, filename):
        '''
        Output the information of all nodes
        :return: None
        '''
        fout = open(filename, 'w')
        line = 'nodename\tcategory\tclusterID\tnumStrongEdges\n'
        fout.write(line)

        fcore_info = open(filename + '_core.txt', 'w')

        for node in self.nodes():
            line = '%s\t%s\t%d\t%d\n' % (node, self.node[node]['category'],self.node[node]['clusterID'],self.node[node]['numStrongEdges'])
            fout.write(line)
            if self.node[node]['category'] == 'core':
                line = '%s\t%d\n' % (node, self.node[node]['clusterID'])
                fcore_info.write(line)
        fout.close()
        fcore_info.close()

    def print_edges(self):
        '''
        Output the SS of al edges
        :return:
        '''
        print('print edges information:')
        ss_values = nx.get_edge_attributes(self, 'SS')
        for edge in self.edges():
            print(edge, "'s SS:", ss_values[edge])

    def get_true_cluster(self, ground_truth_filename):
        '''
        :param ground_truth_filename: the filename in which has the true label of the nodes
        :return:
        '''
        truth_file = open(ground_truth_filename)
        label_truth_dict = dict()
        for line in truth_file:
            '''skip the blank line'''
            if not line.strip():
                continue
            tokens = line.strip().split()
            label_truth_dict[tokens[0]] = int(tokens[1])
        return label_truth_dict


    def calculate_metrics_current(self, label_truth_dict, pred_clusters):
        '''
        Compare the current clusters and the ground truth, calculate the ARI and NMI
        Format in ground truth file:
            node_name_1  clusterID_1
            node_name_2  clusterID_2
                    ... ...
        pred_clusters is a dict(): <clusterID_1, nodes> <clusterID_2, nodes>
        For overlapping case, only one clusterID is used.
        :param label_truth_dict: dict which includes the true labels of nodes
        :param pred_clusters:
        :return: tuple (ARI, NMI)
        '''
        from sklearn import metrics

        label_pred_dict = dict()
        for key in pred_clusters:
            for node in pred_clusters[key]:
                label_pred_dict[node] = key
        ''' only for core points
        if len(label_truth_dict) != len(label_pred_dict):
            print('ERROR! Num of nodes in truth does not equal to num of nodes in pred')
            sys.exit(2)
        '''

        label_truth_list = list()
        label_pred_list = list()
        for node in sorted(label_truth_dict):
            label_truth_list.append(label_truth_dict[node])
            label_pred_list.append(label_pred_dict[node])

        ARI = metrics.adjusted_rand_score(label_truth_list, label_pred_list)
        NMI = metrics.normalized_mutual_info_score(label_truth_list, label_pred_list)
        return (ARI, NMI)

    def calculate_metrics_final(self, label_truth_dict):
        '''
        Compare the final clusters and the ground truth, calculate the ARI and NMI
        Format in ground truth file:
            node_name_1  clusterID_1
            node_name_2  clusterID_2
                    ... ...
        Final clusters are in node['clusterID']
        For overlapping case, only one clusterID is used.
        :param ground_truth_filename:
        :return: tuple (ARI, NMI)
        '''
        from sklearn import metrics

        ''' only for core points
        if len(label_truth_dict) != self.number_of_nodes():
            print('ERROR! Num of nodes in truth does not equal to num of nodes in pred')
            sys.exit(2)
        '''

        label_truth_list = list()
        label_pred_list = list()
        for node in sorted(label_truth_dict):
            label_truth_list.append(label_truth_dict[node])
            label_pred_list.append(self.node[node]['clusterID'])

        ARI = metrics.adjusted_rand_score(label_truth_list, label_pred_list)
        NMI = metrics.normalized_mutual_info_score(label_truth_list, label_pred_list)
        return (ARI, NMI)

    def calculate_modularity_current(self, pred_clusters):
        '''
        Calculate the modularity of the current cluster
        pred_clusters is a dict(): <clusterID_1, nodes> <clusterID_2, nodes>
        For overlapping case, only one clusterID is used.

        :param pred_clusters:
        :return: float, modulariy
        '''
        import community

        label_pred_dict = dict()
        '''cID is used for label noises and hubs'''
        cID = len(pred_clusters)
        for key in pred_clusters:
            if key == -1 or key == -2:
                for node in pred_clusters[key]:
                    label_pred_dict[node] = cID
                    cID += 1
            else:
                for node in pred_clusters[key]:
                    label_pred_dict[node] = key

        if self.number_of_nodes() != len(label_pred_dict):
            print('ERROR! Num of nodes in Graph does not equal to num of nodes in pred')
            sys.exit(2)

        return community.modularity(label_pred_dict, nx.Graph(self))

    def calculate_modularity_final(self):
        '''
        Compare the modularity of the final clusters                     ... ...
        Final clusters are in node['clusterID']
        For overlapping case, only one clusterID is used.

        :return: float modularity
        '''
        import community

        '''cID is used for label noises and hubs'''
        cID = self.number_of_nodes()
        label_pred_dict = dict()

        for node in self.nodes():
            label = self.node[node]['clusterID']
            if label ==-1 or label == -2:
                label_pred_dict[node] = cID
                cID += 1
            else:
                label_pred_dict[node] = label

        if self.number_of_nodes() != len(label_pred_dict):
            print('ERROR! Num of nodes in Graph does not equal to num of nodes in pred')
            sys.exit(2)

        return community.modularity(label_pred_dict, nx.Graph(self))

    import AnytimeSCAN, ast, time
    if __name__ == '__main__':

        if len(sys.argv) < 3:
            print('ERROR!!! Required input: <filename, anytimescan | scan>')
            sys.exit(2)

        my_graph = AnytimeSCAN.AnytimeSCANGraph(sys.argv[1])

        '''for anytime scan'''
        if sys.argv[2] == 'anytimescan':
            if len(sys.argv) < 5:
                print(
                'ERROR!!! Required input: <filename, anytimescan, epsilon(float), mu(int), num_of_AL(int, defaul = 100), isHeuristic(boolean, defaul = True), isRandom_AL(boolean, defaul = False), true_label_filename(string, default = None)>')
                sys.exit(2)

            print('starting anytime_scan')

            if len(sys.argv) == 5:
                my_graph.anytime_scan(sys.argv[1], float(sys.argv[3]), int(sys.argv[4]))
            elif len(sys.argv) == 6:
                my_graph.anytime_scan(sys.argv[1], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
            elif len(sys.argv) == 7:
                my_graph.anytime_scan(sys.argv[1], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                                      ast.literal_eval(sys.argv[6]))
            elif len(sys.argv) == 8:
                my_graph.anytime_scan(sys.argv[1], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                                      ast.literal_eval(sys.argv[6]), ast.literal_eval(sys.argv[7]))
            elif len(sys.argv) >= 9:
                my_graph.anytime_scan(sys.argv[1], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                                      ast.literal_eval(sys.argv[6]), ast.literal_eval(sys.argv[7]), sys.argv[8])

            print('finished anytime_scan')



        '''for original scan'''
        if sys.argv[2] == 'scan':
            if len(sys.argv) < 5:
                print(
                    'ERROR!!! Required input: <filename, scan, epsilon(float), mu(int), true_label_filename(string, default = None)>')
                sys.exit(2)

            print('starting scan')

            if len(sys.argv) == 5:
                my_graph.original_scan(sys.argv[1], float(sys.argv[3]), int(sys.argv[4]))
            elif len(sys.argv) == 6:
                my_graph.original_scan(sys.argv[1], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5])

            print('finished scan')






