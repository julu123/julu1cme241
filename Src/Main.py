from MP import MP

P={
   'C1':({'C2':0.5, 'FB':0.5}),
   'C2':({'C3':0.8,'Sleep':0.2}),
   'C3':({'Pass':0.6,'Pub':0.4}),
   'Pass':({'Sleep':1}),
   'Pub':({'C1':0.2,'C2':0.4,'C3':0.4}),
   'FB':({'C1':0.1,'FB':0.9}),
   'Sleep':({'Sleep':1})
}

test=MP(P)
test.Simulate(steps=10,
              start='C1',
              print_text=True)