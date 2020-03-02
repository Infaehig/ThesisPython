from dolfin import *
from mshr import *

set_log_level(TRACE)

domain = Rectangle(Point(0, 0), Point(5,5))-Rectangle(Point(2,1.25), Point(3,1.75))-Circle(Point(1,4),.25)-Circle(Point(4,4),.25)

info('\nVerbose output of 2d geometry')
info(domain, True)

mesh2d = generate_mesh(domain, 45)
print mesh2d
plot(mesh2d, '2d mesh')

mf = MeshFunction('size_t', mesh2d, 2, mesh2d.domains())
plot(mf)

interactive()
