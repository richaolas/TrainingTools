from turtle import *

s = Shape("compound")
#poly1 = ((0,0),(10,-5),(0,10),(-10,-5))
poly1 = ((0,0),(-10,-5),(10,0),(-10,5))
s.addcomponent(poly1, "red", "blue")
#poly2 = ((0,0),(10,-5),(-10,-5))
#s.addcomponent(poly2, "blue", "red")
register_shape("myshape", s)
shape("myshape")

done()