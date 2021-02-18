
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#initialization of Tensors
x = tf.constant(4.0, shape=(1,1), dtype=tf.float32)
x = tf.constant([[1,2,3],[4,5,6]])
x = tf.ones((3,3))
x = tf.zeros((3,3))
x = tf.eye(3) #create an identity matrix
x = tf.random.normal((3,3), mean=0 , stddev=1)#uniform distribution
x = tf.random.uniform((1,3), minval=0, maxval=1)#uniform with min and max
x = tf.range(start=1, limit=10, delta=2)
###print(x)

#Type casting means chnging datatype
x = tf.cast(x, dtype=tf.float64)
###print(x)

#Now times for mathematicall operations
x = tf.constant([1,2,3])
y = tf.constant([4,5,6])
z = tf.add(x,y) # or z = x + y do same thing

###print(z)
z = tf.subtract(x,y)
###print(z)
z = tf.divide(x,y)# z = x/y
###print(z)
z = tf.multiply(x,y) #z = x*y
###print(z)
#dot product
z = tf.tensordot(x, y , axes=1)
###print(z)
z = tf.reduce_sum(x*y, axis=0)#same as tf.tensordot but change axis
###print(z)
z= x**5
###print(x)

x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x,y)
###print(z)
#or same for matmul is
z = x @ y
###print(z)
#indexing of tensor
x = tf.constant([0,1,2,3,4,5,6,7,8,9])
##print(x[:])
##print(x[::2])#indexing of 1D tensor is same as comnpare to list in python

indices = tf.constant([0,3])
x_ind = tf.gather(x , indices)#we get the indexies value through this 
##print(x_ind)

x = tf.constant([[2,3],
                [4,5],
                [6,7] ])
##print(x[0:2,0:])# here also indexing is simple just add one comma for 2 dim 2 commas for 3 dim

x = tf.range(9)
#print(x)
x = tf.reshape(x,(3,3))
#print(x)

x = tf.transpose(x , perm=[0,1])#[1,0] perms will transpose the matrix means 2nd dim becomes first and 1st dim becomes second
#print(x)