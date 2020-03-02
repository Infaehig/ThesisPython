from dolfin import *
from mshr import *

geom = Rectangle(Point(-1,-1), Point(1,1))
inclusions = Rectangle(Point(-0.5,-0.5),Point(0,0))+Rectangle(Point(0,0), Point(0.5, 0.5))
matrix = geom-inclusions

new = matrix+inclusions
new.set_subdomain(1, matrix)
new.set_subdomain(2, inclusions)

if __name__ == '__main__':
    matrix_mesh = generate_mesh(matrix, 10)
    plot(matrix_mesh, 'matrix')

    inclusions_mesh = generate_mesh(inclusions, 10)
    plot(inclusions_mesh, 'inclusions')

    new_mesh = generate_mesh(new, 10)
    plot(new_mesh, 'new')
    mf = MeshFunction('size_t', new_mesh, 2, new_mesh.domains())
    plot(mf, 'new domains')

    interactive()
