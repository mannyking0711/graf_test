import networkx as nx

from model.V5.scenario import Problem

#Lets suppose we have 5 factories
F2F_time = {
    0: [0,50,180,90,165],
    1: [45,0,65,80,215],
    2: [80,70,0,45,125],
    3: [80,70,40,0,45],
    4: [180,190,95,40,0],
}

#Lets suppose that have m=4 machines per factory, m+1 is the entry/exit point of the factory
M2M_time = {
    0: [0,5,15,12,6],
    1: [7,0,8,21,8],
    2: [8,7,0,11,6],
    3: [11,10,8,0,10],
    4: [6,7,7,9,0],
}

# The transition time (tt_j1_t1_j2_t2) for a j1,t1 that is scheduled in f1/m2 to a dependent task j2/t2 that is scheduled tjo f3/m1 should be
# tt_j1_t1_j2_t2 = m2->m4 + f3->f1 + m4->m1 <=> tt_j1_t1_j2_t2 = 6 + 70 + 7 = 83

class TransportTimeCalculator:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.transportation={}
        FF_matrix = F2F_time
        for f1,factory1 in problem.factories.items():
            F1_MM_matrix = M2M_time
            for m1, machine1 in factory1.machines.items():
                for f2,factory2 in problem.factories.items():
                    F2_MM_matrix = M2M_time
                    for m2, machine2 in factory2.machines.items():
                        self.transportation[(f1,m1,f2,m2)] = (
                            F1_MM_matrix[m1][-1] +
                            FF_matrix[f1][f2] +
                            F2_MM_matrix[-1][m2]
                        )
    def get_transport_time(self,f1:int,m1:int,f2:int,m2:int):
        return self.transportation[(f1,m1,f2,m2)]
    

