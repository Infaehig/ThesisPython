from netgen.csg import *
import dolfin
import numpy as np
import numpy.random as rnd
from dolfin_utils.meshconvert import meshconvert
import os, sys, gc

import logging
import sa_utils
import sa_hdf5

def create_patches(box = np.array([0, 0, 0, 1, 1, 1]), patch_num = 3, patch_nums = None, alpha = 1.25, beta = 2.0, max_resolution = 0.5,
                   num = 6, create_inclusions = False, skip_patches = [], prefix = 'test', logger = None,
                   ldomain = False, corner_refine = 3, hole = False, hole_radius = None, layers = 1, max_refines = 1,
                   elem_per_layer = 3):
    if logger is not None:
        info = logger.info
    else:
        info = print

    basedim = 3
    low = box[:basedim].copy(); high = box[basedim:].copy()
    lengths = high-low
    diameter = np.sqrt(lengths@lengths)
    myeps = sa_utils.myeps*diameter
    center = (high+low)*.5
    info('low {:s}, high {:s}, lengths {:s}, center {:s}, diameter {:.2e}'.format(str(low), str(high), str(lengths), str(center), diameter))

    layer_bricks = []
    layer_hz = lengths[2]/layers
    layer_low = np.array([low[0]-lengths[0], low[1]-lengths[1], low[2]-lengths[2]])
    layer_high = np.array([low[0]+lengths[0], low[1]+lengths[1], low[2]+2*lengths[2]])
    for ii in range(layers-1):
        for jj in range(1,elem_per_layer+1):
            layer_bricks.append(OrthoBrick(Pnt(*layer_low), Pnt(low[0]+lengths[0], low[1]+lengths[1], low[2]+(ii+jj*1./elem_per_layer)*layer_hz)))
        info('layer [{:d}/{:d}], {:s}, {:s}'.format(ii, layers, str(layer_low), str(np.array([low[0]+lengths[0], low[1]+lengths[1], low[2]+ii*layer_hz]))))
    for jj in range(1,elem_per_layer):
        layer_bricks.append(OrthoBrick(Pnt(*layer_low), Pnt(low[0]+lengths[0], low[1]+lengths[1], low[2]+(layers-1+jj*1./elem_per_layer)*layer_hz)))
    layer_bricks.append(OrthoBrick(Pnt(*layer_low), Pnt(*layer_high)))
    sublayers = len(layer_bricks)
    info('layer [{:d}/{:d}], {:s}, {:s}'.format(layers, layers, str(layer_low), str(layer_high)))

    info('{:d} layers, {:d} sublayers, {:d} bricks'.format(layers, sublayers, len(layer_bricks)))

    bc_dict = dict()
    left = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[0], low[0], eps = myeps))
    bc_dict[1] = left
    right = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[0], high[0], eps = myeps))
    bc_dict[2] = right
    front = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[1], low[1], eps = myeps))
    bc_dict[3] = front
    back = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[1], high[1], eps = myeps))
    bc_dict[4] = back
    bottom = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[2], low[2], eps = myeps))
    bc_dict[5] = bottom
    top = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[2], high[2], eps = myeps))
    bc_dict[6] = top
    border = dolfin.AutoSubDomain(lambda xx, on: on)
    if ldomain:
        corner_lr = dolfin.AutoSubDomain(lambda xx, on: on and (xx >= center-myeps).all() and dolfin.near(xx[0], center[0], eps = myeps))
        bc_dict[7] = corner_lr
        corner_fb = dolfin.AutoSubDomain(lambda xx, on: on and (xx >= center-myeps).all() and dolfin.near(xx[1], center[1], eps = myeps))
        bc_dict[8] = corner_fb
        corner_bt = dolfin.AutoSubDomain(lambda xx, on: on and (xx >= center-myeps).all() and dolfin.near(xx[2], center[2], eps = myeps))
        bc_dict[9] = corner_bt
        corner_subdomains = []
        corner_close = 0.1*diameter+myeps
        for ii in range(corner_refine):
            corner_subdomains.append(dolfin.AutoSubDomain((lambda what: lambda xx, on: np.sqrt(xx@xx) < what)(corner_close)))
            corner_close *= 0.5
    if create_inclusions and num:
        info('random inclusions')
        if num:
            number = num*num*num
            info('n = '+str(number))
            inc_radius = 0.5/num
            info('r = '+str(inc_radius))
            nodes = []
            radii = []
            rnd_low = low-0.5*inc_radius
            rnd_high = high+0.5*inc_radius
            width = rnd_high-rnd_low
            while(len(nodes) < number):
                notok = True
                while(notok):
                    new = rnd.rand(3)*width+rnd_low
                    radius = (0.5+rnd.rand())*inc_radius
                    notok = False
                    for old, rr in zip(nodes, radii):
                        diff = new-old
                        if np.sqrt(diff.dot(diff)) < 1.3*(radius+rr):
                            notok = True
                            break
                nodes.append(new.copy())
                radii.append(radius)
            nodes = np.array(nodes)
            radii = np.array(radii)
            info('found locations for '+str(len(nodes))+' inclusions')
            np.savetxt(prefix+'/'+prefix+'_inclusions.csv', np.hstack((nodes, radii.reshape(len(nodes), 1))), fmt = '%.15e', delimiter = ', ')
            del nodes, radii, number, inc_radius

    nohole_whole = OrthoBrick(Pnt(*low), Pnt(*high))
    if ldomain is True:
        nohole_whole = nohole_whole-OrthoBrick(Pnt(*center), Pnt(*(center+2*(high-center))))
    if num:
        data = np.loadtxt(prefix+'/'+prefix+'_inclusions.csv', delimiter = ', ')
        number = len(data)
    else:
        number = 0
    if number:
        nodes = data[:, :3]
        radii = data[:, 3]

        inclusions = Sphere(Pnt(*nodes[0]), radii[0])
        for kk in range(1, len(nodes)):
            inclusions += Sphere(Pnt(*nodes[kk]), radii[kk])
        nohole_matrix = nohole_whole-inclusions
        nohole_incs = nohole_whole*inclusions
    if hole_radius is not None:
        hole = True
    if hole: 
        if hole_radius is None:
            hole_radius = lengths[1]/9.
        near_hole = dolfin.AutoSubDomain(lambda xx, on: on and np.sqrt((xx[0]-center[0])*(xx[0]-center[0])+(xx[1]-center[1])*(xx[1]-center[1])) < hole_radius+1e4*myeps)
        bc_dict[10] = near_hole

    if patch_nums is None:
        hh = lengths[0]/float(patch_num)
        patch_nums = np.array(np.ceil(lengths/hh), dtype = int)
    hs = lengths/patch_nums
    hs_alpha = hs*alpha*0.5
    hs_beta = hs_alpha*beta

    patches = []
    patches_ext = []
    for kk in range(patch_nums[2]):
        pt_z = low[0]+(0.5+kk)*hs[2]
        for jj in range(patch_nums[1]):
            pt_y = low[1]+(0.5+jj)*hs[1]
            for ii in range(patch_nums[0]):
                pt_x = low[2]+(0.5+ii)*hs[0]
                pt_center = np.array([pt_x, pt_y, pt_z])
                pt_low = pt_center-hs_alpha
                if ldomain and (p_low >= center-myeps).all():
                    print('[{:d}, {:d}, {:d}] skipped'.format(ii, jj, kk))
                    continue

                patches.append(OrthoBrick(Pnt(*(pt_center-hs_alpha)), Pnt(*(pt_center+hs_alpha))))
                patches_ext.append(OrthoBrick(Pnt(*(pt_center-hs_beta)), Pnt(*(pt_center+hs_beta)))-patches[-1])
    patch_num = len(patches)
    print('[{:d}] patches total'.format(patch_num))
    patch_fill = int(np.log(patch_num)/np.log(10.))+1

    pt_low = dict()
    pt_high = dict()
    pt_inside = dict()

    info('Patch size computations')
    sa_utils.makedirs_norace(prefix+'/'+prefix+'_patch_descriptors')
    ff = open(prefix+'/'+prefix+'_patch_descriptors/0.csv', 'w')
    ff.write('idx, left, right, front, back, bottom, top\n')
    for kk in range(patch_num):
        info(str(kk+1)+'/'+str(patch_num))
        geo = CSGeometry()
        geo.Add(nohole_whole*patches[kk])
        mesh = geo.GenerateMesh(maxh = max_resolution)
        del geo
        mesh.Export('tmp.msh', 'Gmsh2 Format')
        del mesh
        meshconvert.convert2xml('tmp.msh', 'tmp.xml')
        os.remove('tmp.msh')
        os.remove('tmp_facet_region.xml')
        os.remove('tmp_physical_region.xml')

        mesh = dolfin.Mesh('tmp.xml')
        os.remove('tmp.xml')
        nodes = mesh.coordinates()
        del mesh
        pt_low[kk] = np.min(nodes, axis = 0)
        pt_high[kk] = np.max(nodes, axis = 0)
        ff.write('%d, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n'%(kk, pt_low[kk][0], pt_high[kk][0], pt_low[kk][1], pt_high[kk][1], pt_low[kk][2], pt_high[kk][2]))
        del nodes
        pt_inside[kk] = dolfin.AutoSubDomain(lambda xx, on: (pt_low[kk]-myeps <= xx).all() and (xx <= pt_high[kk]+myeps).all())
    ff.close()
    info('Patch size computations finished')

    hole_ratio = hole_radius/lengths[1]
    for ref in range(max_refines):
        info('Start meshing resolution {:d}/{:d}'.format(ref+1, max_refines))
        res = max_resolution*0.5**ref

        if hole:
            hole_maxh = res*hole_ratio
#           hole_maxh = np.min([res*hole_ratio, lengths[2]/layers])
            if number:
                matrix = nohole_matrix-Cylinder(Pnt(center[0], center[1], center[2]-diameter), Pnt(center[0], center[1], center[2]+diameter), hole_radius).maxh(hole_maxh)
                incs = nohole_matrix-Cylinder(Pnt(center[0], center[1], center[2]-diameter), Pnt(center[0], center[1], center[2]+diameter), hole_radius).maxh(hole_maxh)
            else:
                whole = nohole_whole-Cylinder(Pnt(center[0], center[1], center[2]-diameter), Pnt(center[0], center[1], center[2]+diameter), hole_radius).maxh(hole_maxh)
            
        dirname = '{:s}/{:s}_{:d}_patches/0/'.format(prefix, prefix, ref)
        sa_utils.makedirs_norace(dirname)

        basename = '{:s}/{:s}_{:d}'.format(prefix, prefix, ref)
        info('Global CSG')
        geo = CSGeometry()

        if number:
            geo.Add(matrix*layer_bricks[0])
            for ii in range(1, sublayers):
                geo.Add(matrix*(layer_bricks[ii]-layer_bricks[ii-1]))
            geo.Add(incs*layer_bricks[0])
            for ii in range(1, sublayers):
                geo.Add(incs*(layer_bricks[ii]-layer_bricks[ii-1]))
        else:
            geo.Add(whole*layer_bricks[0])
            for ii in range(1, sublayers):
                geo.Add(whole*(layer_bricks[ii]-layer_bricks[ii-1]))
        info('Global CSG constructed')
        mesh = geo.GenerateMesh(maxh = res)
        info('Global surface meshed')
        del geo

        gc.collect()

        mesh.GenerateVolumeMesh()
        mesh.Export(basename+'.msh', 'Gmsh2 Format')
        meshconvert.convert2xml(basename+'.msh', basename+'.xml')
        del mesh
        os.remove(basename+'.msh')
        os.remove(basename+'_facet_region.xml')
        info('Global volume meshed')

        gc.collect()

        global_mesh = dolfin.Mesh(basename+'.xml')
        tmp_nodes = global_mesh.coordinates()
        tmp_low = np.min(tmp_nodes, axis = 0)
        tmp_high = np.max(tmp_nodes, axis = 0)
        info('global mesh: {:s}, {:s}, {:s}'.format(str(tmp_low), str(tmp_high), str(top.inside(tmp_high, True))))

        os.remove(basename+'.xml')
        info('Correcting cell markers')
        global_domains_tmp = dolfin.MeshFunction('size_t', global_mesh, basename+'_physical_region.xml')
        os.remove(basename+'_physical_region.xml')
        global_domains_tmp.array()[:] -= np.min(global_domains_tmp.array())
        global_domains_tmp.array()[:] //= elem_per_layer
        global_domains = dolfin.MeshFunction('size_t', global_mesh, basedim, 0)
        if number:
            where = np.where(global_domains_tmp.array() < layers)
            global_domains.array()[where] = 4*global_domains_tmp.array()[where]
            where = np.where(layers <= global_domains_tmp.array())
            global_domains.array()[where] = 4*(global_domains_tmp.array()[where]-layers)+1
            del where
        else:
            global_domains.array()[:] = 4*global_domains_tmp.array()
        del global_domains_tmp
        if ldomain:
            for ii in range(corner_refine):
                mf = dolfin.CellFunction('bool', global_mesh, False)
                corner_subdomains[ii].mark(mf, True)
                global_mesh = dolfin.refine(global_mesh, mf)
                global_domains = dolfin.adapt(global_domains, global_mesh)
                del mf
        inside_fun = dolfin.MeshFunction('bool', global_mesh, basedim, True)
        global_mesh = dolfin.refine(global_mesh, inside_fun)
        global_domains = dolfin.adapt(global_domains, global_mesh)
        info('Correcting facet markers')
        global_facets = dolfin.MeshFunction('size_t', global_mesh, basedim-1, 0)
        for key in bc_dict:
            bc_dict[key].mark(global_facets, key)
        sa_hdf5.write_dolfin_mesh(global_mesh, basename, cell_function = global_domains, facet_function = global_facets)
        del global_facets, global_mesh, global_domains, basename

        gc.collect()

        for kk in range(patch_num):
            if kk in skip_patches:
                continue
            info(str(kk)+'/'+str(patch_num))
            basename = 'patch_'+str(kk).zfill(patch_fill)
            extname = basename+'_'+str(beta)

            info('    csg')
            geo = CSGeometry()
            if number:
                geo.Add(matrix*layer_bricks[0]*patches[kk])
                for ii in range(1, sublayers):
                    geo.Add(matrix*(layer_bricks[ii]-layer_bricks[ii-1])*patches[kk])
                geo.Add(incs*layer_bricks[0]*patches[kk])
                for ii in range(1, sublayers):
                    geo.Add(incs*(layer_bricks[ii]-layer_bricks[ii-1])*patches[kk])
                geo.Add(matrix*layer_bricks[0]*patches_ext[kk])
                for ii in range(1, sublayers):
                    geo.Add(matrix*(layer_bricks[ii]-layer_bricks[ii-1])*patches_ext[kk])
                geo.Add(incs*layer_bricks[0]*patches_ext[kk])
                for ii in range(1, sublayers):
                    geo.Add(incs*(layer_bricks[ii]-layer_bricks[ii-1])*patches_ext[kk])
            else:
                geo.Add(whole*layer_bricks[0]*patches[kk])
                for ii in range(1, sublayers):
                    geo.Add(whole*(layer_bricks[ii]-layer_bricks[ii-1])*patches[kk])
                geo.Add(whole*layer_bricks[0]*patches_ext[kk])
                for ii in range(1, sublayers):
                    geo.Add(whole*(layer_bricks[ii]-layer_bricks[ii-1])*patches_ext[kk])
            info('    csg done')
            mesh = geo.GenerateMesh(maxh = res)
            info('    surface meshed')
            del geo

            gc.collect()

            mesh.GenerateVolumeMesh()
            info('    volume meshed')
            mesh.Export(dirname+'/'+basename+'.msh', 'Gmsh2 Format')
            meshconvert.convert2xml(dirname+'/'+basename+'.msh', dirname+'/'+basename+'.xml')
            del mesh
            os.remove(dirname+'/'+basename+'.msh')
            os.remove(dirname+'/'+basename+'_facet_region.xml')

            gc.collect()

            ext_mesh = dolfin.Mesh(dirname+'/'+basename+'.xml')
            os.remove(dirname+'/'+basename+'.xml')
            info('    cell function')
            ext_domains_tmp = dolfin.MeshFunction('size_t', ext_mesh, dirname+'/'+basename+'_physical_region.xml')
            os.remove(dirname+'/'+basename+'_physical_region.xml')
            ext_domains_tmp.array()[:] -= np.min(ext_domains_tmp.array())
            ext_domains_tmp.array()[:] //= elem_per_layer
            ext_domains = dolfin.MeshFunction('size_t', ext_mesh, basedim, 0)
            if number:
                where = np.where(ext_domains_tmp.array() < layers)
                ext_domains.array()[where] = 4*ext_domains_tmp.array()[where]
                where = np.where(layers <= ext_domains_tmp.array() < 2*layers)
                ext_domains.array()[where] = 4*(ext_domains_tmp.array()[where]-layers)+1
                where = np.where(2*layers <= ext_domains_tmp.array() < 3*layers)
                ext_domains.array()[where] = 4*(ext_domains_tmp.array()[where]-2*layers)+2
                where = np.where(3*layers <= ext_domains_tmp.array())
                ext_domains.array()[where] = 4*(ext_domains_tmp.array()[where]-3*layers)+3
                del where
            else:
                where = np.where(ext_domains_tmp.array() < layers)
                ext_domains.array()[where] = 4*ext_domains_tmp.array()[where]
                where = np.where(layers <= ext_domains_tmp.array())
                ext_domains.array()[where] = 4*(ext_domains_tmp.array()[where]-layers)+2
            del ext_domains_tmp
            if ldomain:
                for ii in range(corner_refine):
                    mf = dolfin.CellFunction('bool', ext_mesh, False)
                    corner_subdomains[ii].mark(mf, True)
                    ext_mesh = dolfin.refine(ext_mesh, mf)
                    ext_domains = dolfin.adapt(ext_domains, ext_mesh)
                    del mf
            inside_fun = dolfin.MeshFunction('bool', ext_mesh, basedim, False)
            pt_inside[kk].mark(inside_fun, True)
            ext_mesh = dolfin.refine(ext_mesh, inside_fun)
            del inside_fun
            ext_domains = dolfin.adapt(ext_domains, ext_mesh)
            info('    cell function done')
         
            pt_mesh = dolfin.SubMesh(ext_mesh, pt_inside[kk])

            pt_domains = dolfin.MeshFunction('size_t', pt_mesh, basedim, 0)
            tree = ext_mesh.bounding_box_tree()
            for cell in dolfin.cells(pt_mesh):
                global_index = tree.compute_first_entity_collision(cell.midpoint())
                pt_domains[cell] = ext_domains[dolfin.Cell(ext_mesh, global_index)]
            del tree

            pt_facets = dolfin.MeshFunction('size_t', pt_mesh, basedim-1, 0)
            border.mark(pt_facets, 100)
            for key in bc_dict:
                bc_dict[key].mark(pt_facets, key)
            sa_hdf5.write_dolfin_mesh(pt_mesh, '{:s}/{:s}'.format(dirname, basename), cell_function = pt_domains, facet_function = pt_facets)

            tmp_nodes = ext_mesh.coordinates()
            tmp_low = np.min(tmp_nodes, axis = 0)
            tmp_high = np.max(tmp_nodes, axis = 0)
            is_part = (tmp_low > low+myeps).any() or (tmp_high < high-myeps).any()
            del tmp_low, tmp_high, tmp_nodes
            info('patch [{:d}/{:d}], beta [{:.2e}] is real subdomain [{:}]'.format(kk+1, patch_num, beta, is_part))

            if is_part:
                vals = np.arange(1,11)
            else:
                vals = np.unique(pt_facets.array())
                vals = vals[np.where(vals > 0)]
            patch_dict = dict()
            for key in bc_dict:
                if key in vals:
                    patch_dict[key] = bc_dict[key]
                else:
                    patch_dict[key] = dolfin.AutoSubDomain((lambda what: (lambda xx, on: bc_dict[what].inside(xx, on) and pt_inside[kk].inside(xx, on)))(key))
            ext_facets = dolfin.MeshFunction('size_t', ext_mesh, basedim-1, 0)
            border.mark(ext_facets, 100)
            for key in patch_dict:
                patch_dict[key].mark(ext_facets, key)
            del patch_dict, vals
            sa_hdf5.write_dolfin_mesh(ext_mesh, dirname+'/'+basename+'_'+str(beta), cell_function = ext_domains, facet_function = ext_facets)
            del ext_mesh, ext_domains, ext_facets, pt_mesh, pt_domains, pt_facets

            gc.collect()

    del pt_low, pt_high, pt_inside
