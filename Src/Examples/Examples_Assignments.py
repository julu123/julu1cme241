from scipy.stats import poisson
from MDP_B import MDP_B


class Jack_car(MDP_B):
    def __init__(self,statex:int=20,statey:int=20,move_max:int=5,gamma:float=0.99,reward:float=10,lambdx:float=4,lambdy:float=3,retx:float=2,rety:float=3):
        self.lx=lambdx
        self.ly=lambdy
        self.reward=reward
        self.statex=statex
        self.statey=statey
        self.retx=retx
        self.rety=rety
        self.move_max=move_max
        self.gamma=gamma
        MDP_B.__init__(self,self.Create_States(),self.gamma)
        
    def Create_Outcomes(self,x,y):
        xprobs = [poisson.pmf(i,self.lx) for i in (range(x))] #demand
        xprobs.append(1-sum(xprobs))
        yprobs = [poisson.pmf(i,self.ly) for i in (range(y))] #demand
        yprobs.append(1-sum(yprobs))
        #xdemand = [[i,xprobs[i]] for i in range(x+1)]
        #ydemand = [[i,yprobs[i]] for i in range(y+1)]
        #demand = [[(x-i,y-j,xprobs[i]*yprobs[j],self.reward*(i+j))] for i in range(x+1) for j in range(y+1)]
        #xrets = [[min(i,self.statex),poisson.pmf(i,self.retx)] for i in range(self.statex+1)] #cars returning
        #yrets = [[min(i,self.statey),poisson.pmf(i,self.rety)] for i in range(self.statey+1)] #cars returning
        #return(xrets)
        return({(min(x+self.retx,self.statex)-i,min(y+self.rety,self.statey)-j):(xprobs[i]*yprobs[j],self.reward*i+self.reward*j) for i in range(x+1) for j in range(y+1)})
    
    def Create_Actions(self,x,y):
        xlow=x-self.move_max
        ylow=y-self.move_max
        if ylow >= 0 and xlow >= 0:
            a=([(x-self.move_max+i,y+self.move_max-i) for i in range(2*self.move_max+1)])
        elif ylow < 0 and xlow >=0: 
            c=y+self.move_max
            a=([(x-5+i,y+5-i) for i in range(1+c)])
        elif ylow >= 0 and xlow <0:
            c=x+self.move_max
            a=([(i,y+x-i) for i in range(1+c)])
        else:
            c=x+y
            a=([(i,y+x-i) for i in range(1+c)])
        q=[]
        for i in a:
            if i[0] <= self.statex and i[1] <= self.statey:
                q.append(i)
        test={str(i[1]-y)+'m':self.Create_Outcomes(i[0],i[1]) for i in q}
        return(test)
    
    def Create_States(self):
        return({(i,j):self.Create_Actions(i,j) for i in range(self.statex+1) for j in range(self.statey+1)})
     
    
    
      
class Gambler(MDP_B):
    def __init__(self,ph:float=0.4,max_capital:float=100,gamma:float=0.99):
        self.gamma=gamma
        self.ph=ph
        self.max_capital=max_capital
        MDP_B.__init__(self,self.Create_States(),self.gamma)
        
    def Create_Outcome(self,state,action):
        if action != 0:
            return{str(state+action):(self.ph,action),str(state-action):((1-self.ph),-action)}
        else:
            return{str(state):(1,action if state != self.max_capital else 1)}
    
    def Create_Actions(self,state):
        return{str(i):self.Create_Outcome(state,i) for i in range(min(state,self.max_capital-state)+1)}
    
    def Create_States(self):
        return{str(i):self.Create_Actions(i) for i in range(self.max_capital+1)}

      
      
class Grid_World(MDP_B):
    def __init__(self,gamma:float=0.99):
      self.gammma=gamma
      MDP_B.__init__(self,self.Get_info(),self.gamma)
      
    def Get_info(self):
        test={str(10*(i+1)+j+1):0 for i in range(5) for j in range(5)}
        for i in test:
            Up = int(i) if int(i)<=15 else int(i)-10
            Left = int(i) if int(i)%10 == 1 else int(i)-1
            Right = int(i) if int(i)%10 == 5 else int(i)+1
            Down = int(i) if int(i)>=51 else int(i)+10
            RU = -1 if int(i) <= 15 else 0
            RL = -1 if int(i)%10 == 1 else 0
            RR = -1 if int(i)%10 == 5 else 0
            RD = -1 if int(i)>=51 else 0
            if int(i) == 12:
                Up = 52
                Left = Up
                Right = Up
                Down = Up
                RU = 10
                RL = RU
                RR = RU
                RD = RU
            if int(i)==14:
                Up = 34
                Left = Up
                Right = Up
                Down = Up
                RU = 5
                RL = RU
                RR = RU
                RD = RU
            test[i]={'Up':({str(Up):1},RU),
                    'Left':({str(Left):1},RL),
                    'Right':({str(Right):1},RR),
                    'Down':({str(Down):1},RD)
                   }
        return test


