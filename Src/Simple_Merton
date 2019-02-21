#!/usr/bin/env python
# coding: utf-8

class Simple_Merton(MDP_A):
    def __init__(self,risk_aversion:float,mu:float,sigma:float,r:float,wealth:(float or int),steps:int,discount:float):
        self.a = risk_aversion
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.w = wealth
        self.gamma = discount
        self.steps = steps
        self.all_info = self.Get_all_info()
        
    def Create_binomal_outcome(self,wealth,pi,p:float=1/2):
        s = self.sigma
        mu = self.mu
        r = self.r
        w = wealth
        up = (p*w*pi*(mu-r)+np.sqrt(-(p-1)*p*pi**2*s**2*w**2)+p*w*r+p*w)/p
        down = (p*w*pi*(mu-r)-w*pi*(mu-r)+np.sqrt(-(p-1)*p*pi**2*s**2*w**2)+p*w*r+p*w-r*w-w)/(p-1)
        #return {'u':(p,round(up)),'d':((1-p),round(down))}
        #print({round(up):p,round(down):1-p})
        if round(up) == round(down):
            return {int(round(up)):1}
        else:
            return {min(400,int(round(up))):p,max(int(round(down)),-200):1-p}

    def Utility_Function(self,wealth,a):
        return wealth**(1-a)/(1-a)

    def Create_actions(self,wealth,wealth_partitions:int=10,pi_space:list=[-2, 2],pi_partition:int=5):
        a = self.a
        mu = self.mu
        sigma = self.sigma
        r = self.r
        return {(w,pi/pi_partition):(self.Create_binomal_outcome(w,pi/pi_partition),self.Utility_Function(wealth-w,a))
                    for pi in range(pi_space[0]*pi_partition,(pi_space[1])*pi_partition+1) 
                        for w in range(0,wealth+int(wealth/wealth_partitions),max(int(wealth/wealth_partitions),1))}
    
    def Get_all_info(self,min_wealth:int=-200,max_wealth:int=400):
        matrix= {i:self.Create_actions(i) for i in range(min_wealth,max_wealth)}
        return MDP_A(matrix)

