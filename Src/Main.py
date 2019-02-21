from MRP_A import MRP_A

P={
   'C1':({'C2':0.5, 'FB':0.5},-2),
   'C2':({'C3':0.8,'Sleep':0.2},-2),
   'C3':({'Pass':0.6,'Pub':0.4},-2),
   'Pass':({'Sleep':1},10),
   'Pub':({'C1':0.2,'C2':0.4,'C3':0.4},1),
   'FB':({'C1':0.1,'FB':0.9},-1),
   'Sleep':({'Sleep':1},0)
}

test=MRP_A(P)
test.Simulate_Rewards(steps=10,start='Pub',print_text=True)