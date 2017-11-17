# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:27:03 2017

@author: mkundu
"""

## IF,Else,elif Statement ##

## This is the code structure of Python , if elif and else ##
# if case1:
    #perform action1
# elif case2:
    #perform action2
# else:
    #perform action3


##### Live example ####

x = False

if x: ### that means x is nothing 
    print 'x is true'
else:
    print 'not true'

x = True

if x: ### that means x is nothing 
    print 'x is true'
else:
    print 'not true'
    
loc = 'bank'

if loc == 'auto':
    print 'bhag'
elif loc == 'bank':
    print 'a ja'
elif loc == 'gaon':
    print 'bhag'
else:
    print 'ho kaha '
    

 #### for Loops -- we can use for iterate on ###

l = [1,2,3,4,5]

for num in l:
    print num

#MODULA (% ie 10 % 7 = 3 it gives reminder)

## Check where in the list how many elements are divisible by 2 

# even
for num in l:
    if num % 2 == 0:
        print num
#odd
for num in l:
    if num % 2 == 1:
        print num


# both combination 
for num in l:
    if num % 2 == 0:
        print num 
    elif num % 2 == 1:
        print 'Odd hai bhai !'
        
        
# Addition of Elements in the list 
list_sum = 0

for num in l:
    list_sum = num + list_sum ## assignment 
    
print list_sum
    
    

## Tuple in List pairs 

l = [(2,4), (6,8), (10,12)]

l[0][0] ## first element of the first tuple set from a list 

for tup in l:
    print tup 

#1st element of Tuples 
for (t1,t2) in l:
    print t1
# 2nd element of Tuples    
for (t1,t2) in l:
    print t2
# any operation we can do 
for (t1,t2) in l:
    
    print t1+t2


##### iterate through  a dictonory ###

d = {'k1':1, 'k2':2, 'k3':3}

for item in d:
    print item ## only prints the KEYS 
    

#### if the iter the both keys and values  ### 
for k,v in d.iteritems():
    print (k,v)

for k,v in d.iteritems():
    print k
    print v
           

######### While Loops ##########

x = 0

while x < 10:
    print 'x is currently:', x 
    x += 1


## WHile & ELse ##    
x = 0

while x < 10:
    print 'x is currently:', x 
    x += 1
else:
    print 'All is done'

## break - break the current closet enclosing loop 
## continue - go to the top of the closest enclosing loop 
## pass - Does anything at all 

### Structure of the code ###
#while test: 
    #code statement
    #if test:
        #break
    #if test:
        #continue 
#else:

# while + nested if and else     
x = 0

while x < 10:
    print 'x is currently', x
    x += 1
    if x == 3:
        print 'x is equals 3'
    else:
        print 'continue'
        continue 

# same as above + break #
x = 0
while x < 10:
    print 'x is currently', x
    x += 1
    if x == 3:
        print 'x is equals 3'
        break 
    else:
        print 'continue' #### this peice of code will not executed ###
        continue

## beware the about INFINITE LOOPS ###
 while True:
     print 'he he '
 
    








