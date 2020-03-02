from dolfin import *

comm = MPI.comm_world
rank = MPI.rank(comm)
rank_string = str(rank)
size = MPI.size(comm)

import numpy as np

import multiprocessing
import logging, logging.handlers
import inspect
import time, os, gc, sys
import ctypes

myeps = 1e-10

def human_time(tt):
    seconds = tt % 60
    if seconds > 0.01:
        ret = '{:.2f}s'.format(seconds)
    else:
        ret = '{:.2e}s'.format(seconds)
    tt = int((tt-seconds))//60
    if tt == 0:
        return ret
    minutes = tt % 60
    ret = '{:d}m:{:s}'.format(minutes, ret)
    tt = (tt-minutes)//60
    if tt == 0:
        return ret
    hours = tt % 24
    ret = '{:d}h:{:s}'.format(hours, ret)
    tt = (tt-hours)//24
    if tt == 0:
        return ret
    return '{:d}d:{:s}'.format(tt, ret)

def makedirs_norace(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok = True)

def get_shared_array(shape):
    base = multiprocessing.Array(ctypes.c_double, int(np.prod(shape)))
    return np.frombuffer(base.get_obj()).reshape(shape)

class IndentFormatter(logging.Formatter):
    def __init__(self, **args):
        logging.Formatter.__init__(self, **args)
        self.baseline = len(inspect.stack())
    def format(self, rec):
        stack = inspect.stack()
        rec.indent = '. '*(len(stack)-self.baseline)
        rec.function = stack[8][3]
        out = logging.Formatter.format(self, rec)
        del rec.indent; del rec.function
        return out

formatter = IndentFormatter(fmt = '[%(asctime)s]-[%(processName)s]\n%(indent)s%(message)s')

class LogWrapper:
    def __init__(self, logname, *, stdout=False):
        self.lock = multiprocessing.Lock()
        self.lock.acquire()
        self.logger = logging.getLogger(rank_string)
        self.logger.setLevel(logging.INFO)
        path = os.path.dirname(logname)
        makedirs_norace(path)
        self.handler = logging.FileHandler(logname+'_rk_'+str(rank)+'.log', mode = 'w')
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        if stdout:
            self.stdout_handler = logging.StreamHandler(sys.stdout)
            self.stdout_handler.setLevel(logging.INFO)
            self.stdout_handler.setFormatter(formatter)
            self.logger.addHandler(self.stdout_handler)
        self.lock.release()

    def info(self, *args, **kwargs):
        self.lock.acquire()
        self.logger.info(*args, **kwargs)
        self.lock.release()

    def critical(self, *args, **kwargs):
        self.lock.acquire()
        self.logger.critical(*args, **kwargs)
        self.lock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lock.acquire()
        self.logger.handlers = []
        del self.logger
        self.handler.flush()
        self.handler.close()
        del self.handler
        self.lock.release()
        del self.lock

def get_time():
    return time.process_time()


def simple_batch_fun(uus, VV, low = 0, high = 1, fun = None, logger = None, max_procs = None):
    uus_len = high-low
    cpu_count = multiprocessing.cpu_count() if max_procs is None else np.min([multiprocessing.cpu_count(), max_procs])
    if cpu_count > 1:
        done_queue = multiprocessing.Queue()
        def done_put(qq):
            qq.put(None)
        def done_get():
            done_queue.get()
    else:
        done_queue = None
        def done_put(qq):
            pass
        def done_get():
            pass

    if logger is not None:
        logger.info('SIMPLE: trying to allocate shared size [{:d}] array'.format(uus_len*VV.dim()))
        polys_array = get_shared_array((uus_len, VV.dim()))
        logger.info('SIMPLE: shared array allocated')
        def tmp_fun(block_low, block_high, qq):
            logger.info('    SIMPLE TMP: ['+str(block_low)+', '+str(block_high)+'] start')
            for ii in range(block_low, block_high):
                logger.info('        [{:d}/{:d}] start'.format(ii+1, high))
                polys_array[ii-low] = fun(uus[ii]).vector().get_local()
                logger.info('        [{:d}/{:d}] end'.format(ii+1, high))
            logger.info('    SIMPLE TMP: ['+str(block_low)+', '+str(block_high)+'] computations done')
            done_put(qq)
            logger.info('    SIMPLE TMP: ['+str(block_low)+', '+str(block_high)+'] queue put')
        processes = []
        logger.info('SIMPLE: Using [{:d}] processes with [{:d}] requested and [{:d}] cpus available'.format(cpu_count, max_procs, multiprocessing.cpu_count()))
        if cpu_count > 1:
            block = int(np.ceil(uus_len / cpu_count))
            logger.info('SIMPLE: ['+str(block)+'] for ['+str(cpu_count)+']cpus for ['+str(uus_len)+'] functions')
            block_low = low
            block_high = np.min([block_low+block, high])
            for cpu in range(cpu_count):
                if block_low >= block_high:
                    break
                logger.info('SIMPLE: ['+str(cpu)+']: ['+str(low)+', '+str(block_low)+', '+str(block_high)+', '+str(high)+'] starting')
                processes.append(multiprocessing.Process(target = tmp_fun, args = (block_low, block_high, done_queue)))
                processes[-1].start()
                block_low = block_high
                block_high = np.min([block_high+block, high])
                logger.info('SIMPLE: ['+str(cpu)+']: ['+str(low)+', '+str(block_low)+', '+str(block_high)+', '+str(high)+'] started')
            proc_count = len(processes)
            for cpu, proc in enumerate(processes):
                done_get()
                logger.info('SIMPLE: ['+str(cpu+1)+'/'+str(proc_count)+'] fetched')
            for cpu, proc in enumerate(processes):
                proc.join()
                proc = None
                processes[cpu] = None
                logger.info('SIMPLE: ['+str(cpu+1)+'/'+str(proc_count)+'] joined')
            del processes
        else:
            tmp_fun(low, high, done_queue)
            done_get()
        del done_queue
        polys = []
        for ii in range(uus_len):
            polys.append(Function(VV, name = 'u'))
            polys[-1].vector().set_local(polys_array[ii])
        del polys_array
        logger.info('SIMPLE: END')
    else:
        polys_array = get_shared_array((uus_len, VV.dim()))
        def tmp_fun(block_low, block_high, qq):
            for ii in range(block_low, block_high):
                polys_array[ii-low] = fun(uus[ii]).vector().get_local()
            done_put(qq)
        processes = []
        if cpu_count > 1:
            block = int(np.ceil(uus_len / cpu_count))
            block_low = low
            block_high = np.min([block_low+block, high])
            for cpu in range(cpu_count):
                if block_low >= block_high:
                    break
                processes.append(multiprocessing.Process(target = tmp_fun, args = (block_low, block_high, done_queue)))
                processes[-1].start()
                block_low = block_high
                block_high = np.min([block_high+block, high])
            proc_count = len(processes)
            for cpu, proc in enumerate(processes):
                done_get()
            for cpu, proc in enumerate(processes):
                proc.join()
                proc = None
                processes[cpu] = None
            del processes
        else:
            tmp_fun(low, high, done_queue)
            done_get()
        del done_queue
        polys = []
        for ii in range(uus_len):
            polys.append(Function(VV, name = 'u'))
            polys[-1].vector().set_local(polys_array[ii])
        del polys_array
    return polys

def batch_fun(uus, VV, low = 0, high = 1, offsets = None, fun = None, polys = None, times = None, logger = None, max_procs = None):
    cpu_count = multiprocessing.cpu_count() if max_procs is None else np.min([multiprocessing.cpu_count(), max_procs])
    offset = offsets[low]
    num_polys = offsets[high]-offset
    if cpu_count > 1:
        done_queue = multiprocessing.Queue()
        def done_put(qq):
            qq.put(None)
        def done_get():
            done_queue.get()
    else:
        done_queue = None
        def done_put(qq):
            pass
        def done_get():
            pass
    if logger is not None:
        logger.info('BATCH: trying to allocate shared size [{:d}] array'.format(num_polys*VV.dim()))
        polys_array = get_shared_array((num_polys, VV.dim()))
        logger.info('BATCH: shared array allocated')
        times_array = get_shared_array(offsets[high])
        def tmp_fun(block_low, block_high, qq):
            logger.info('    BATCH TMP: ['+str(block_low)+', '+str(block_high)+'] start')
            for ii in range(block_low, block_high):
                logger.info('        [{:d}/{:d}] start'.format(ii+1, offsets[high]))
                old = get_time()
                polys_array[ii-offset] = fun(uus[ii]).vector().get_local()
                new = get_time()
                times_array[ii] = new-old
                logger.info('        [{:d}/{:d}] end'.format(ii+1, offsets[high]))
            logger.info('    BATCH TMP: ['+str(block_low)+', '+str(block_high)+'] computations done')
            done_put(qq)
            logger.info('    BATCH TMP: ['+str(block_low)+', '+str(block_high)+'] times put in queue')
        processes = []
        logger.info('BATCH: Using [{:d}] processes with [{:d}] requested and [{:d}] cpus available'.format(cpu_count, max_procs, multiprocessing.cpu_count()))
        if cpu_count > 1:
            block = int(np.ceil(num_polys / cpu_count))
            logger.info('BATCH: blocks of ['+str(block)+'] for ['+str(cpu_count)+'] cpus for ['+str(num_polys)+'] functions')
            block_low = offsets[low]
            block_high = np.min([block_low+block, offsets[high]])
            for cpu in range(cpu_count):
                if block_low >= block_high:
                    break
                logger.info('BATCH: ['+str(cpu)+']: ['+str(offsets[low])+', '+str(block_low)+', '+str(block_high)+', '+str(offsets[high])+'] starting')
                processes.append(multiprocessing.Process(target = tmp_fun, args = (block_low, block_high, done_queue)))
                processes[-1].start()
                block_low = block_high
                block_high = np.min([block_high+block, offsets[high]])
                logger.info('BATCH: ['+str(cpu)+']: ['+str(offsets[low])+', '+str(block_low)+', '+str(block_high)+', '+str(offsets[high])+'] started')
            proc_count = len(processes)
            logger.info('BATCH: ['+str(proc_count)+'] processes started')
            for cpu, proc in enumerate(processes):
                done_get()
                logger.info('BATCH: ['+str(cpu+1)+'/'+str(proc_count)+'] fetched')
            for cpu, proc in enumerate(processes):
                proc.join()
                proc = None
                processes[cpu] = None
                logger.info('BATCH: ['+str(cpu+1)+'/'+str(proc_count)+'] joined')
            del processes
        else:
            tmp_fun(offsets[low], offsets[high], done_queue)
            done_get()
        del done_queue
        for ii in range(num_polys):
            polys.append(Function(VV, name = 'u'))
            polys[-1].vector().set_local(polys_array[ii])
        del polys_array
        offset = offsets[low]
        for deg in range(low, high):
            tmp = times[-1]
            for ii in range(offsets[deg], offsets[deg+1]):
                tmp += times_array[ii]
            times.append(tmp)
        del times_array
        logger.info('BATCH: END')
    else:
        polys_array = get_shared_array((num_polys, VV.dim()))
        times_array = get_shared_array(offsets[high])
        def tmp_fun(block_low, block_high, qq):
            for ii in range(block_low, block_high):
                old = get_time()
                polys_array[ii-offset] = fun(uus[ii]).vector().get_local()
                new = get_time()
                times_array[ii] = new-old
            done_put(qq)
        processes = []
        cpu_count = multiprocessing.cpu_count() if max_procs is None else np.min([multiprocessing.cpu_count(), max_procs])
        if cpu_count > 1:
            block = int(np.ceil(num_polys / cpu_count))
            block_low = offsets[low]
            block_high = np.min([block_low+block, offsets[high]])
            for cpu in range(cpu_count):
                if block_low >= block_high:
                    break
                processes.append(multiprocessing.Process(target = tmp_fun, args = (block_low, block_high, done_queue)))
                processes[-1].start()
                block_low = block_high
                block_high = np.min([block_high+block, offsets[high]])
            proc_count = len(processes)
            for cpu, proc in enumerate(processes):
                done_get()
            for cpu, proc in enumerate(processes):
                proc.join()
                proc = None
                processes[cpu] = None
            del processes
        else:
            tmp_fun(offsets[low], offsets[high], done_queue)
            done_get()
        del done_queue
        for ii in range(num_polys):
            polys.append(Function(VV, name = 'u'))
            polys[-1].vector().set_local(polys_array[ii])
        del polys_array
        offset = offsets[low]
        for deg in range(low, high):
            tmp = times[-1]
            for ii in range(offsets[deg], offsets[deg+1]):
                tmp += times_array[ii]
            times.append(tmp)
        del times_array

def matplotlib_stuff():
    import matplotlib.pyplot as plt

    ploteps = 5e-2 
    legendx = 3.6
    legendy = 1.2

    line_styles = ['-', ':', '--', '-.']
    num_line_styles = len(line_styles)
    marker_styles = ['o', '+', 's', 'x', 'p', '*', 'd']
    num_marker_styles = len(marker_styles)
    styles = [aa+bb for aa in marker_styles for bb in line_styles]
    num_styles = len(styles)
    color_styles = ['r', 'g', 'b', 'c', 'y', 'm', 'k', '#a0522d', '#6b8e23']
    num_color_styles = len(color_styles)

    def set_log_ticks(ax, minval, maxval, xaxis = False, semilog = False):
        if semilog:
            fac = ploteps*(maxval-minval)
            minval = 0
    #       minval -= fac;
            maxval += fac;
            if maxval % 20:
                maxval = 20*np.ceil(maxval/20)
            if xaxis:
                ax.set_xlim(minval, maxval)
                ticks = ax.get_xticks()
                ax.set_xticklabels(r'$'+str(kk).rstrip('0').rstrip('.')+r'$' for kk in ticks)
            else:
                ax.set_ylim(minval, maxval)
                ticks = ax.get_yticks()
                ax.set_yticklabels(r'$'+str(kk)+r'$' for kk in ticks)
        else:
            fac = np.exp(ploteps*np.log(maxval/minval));
            minval /= fac; maxval *= fac

            low_pow = int(np.floor(np.log(minval)/np.log(10)))
            low_mul = int(np.floor(minval/10**low_pow))
            low = low_mul*10**low_pow
            top_pow = int(np.floor(np.log(maxval)/np.log(10)))
            top_mul = int(np.floor(maxval/10**top_pow))
            if top_mul*10**top_pow < maxval:
                if top_mul == 9:
                    top_pow += 1
                    top_mul = 1
                else:
                    top_mul += 1
            top = top_mul*10**top_pow
            inter = range(low_pow+1, top_pow+(1 if top_mul > 1 else 0))
            if len(inter):
                ticks = [10**kk for kk in inter]
                labels = [r'$10^{'+str(kk)+r'}$' for kk in inter]
                width = np.log(ticks[-1]/ticks[0])
            else:
                ticks = []; labels = []
            if not len(inter) or (np.log(minval/low) < .1*width and
                                  np.log(ticks[0]/low) > .2*width):
                ticks = [low]+ticks
                if low_mul > 1:
                    label = r'$'+str(low_mul)+r'\cdot 10^{'+str(low_pow)+r'}$'
                else:
                    label = r'$10^{'+str(low_pow)+r'}$'
                labels = [label]+labels
            if not len(inter) or (np.log(top/maxval) < .1*width and
                                  np.log(top/ticks[-1]) > .2*width):
                ticks = ticks+[top]
                if top_mul > 1:
                    label = r'$'+str(top_mul)+r'\cdot 10^{'+str(top_pow)+r'}$'
                else:
                    label = r'$10^{'+str(top_pow)+r'}$'
                labels = labels+[label]
            minval = np.min([minval, ticks[0]])
            maxval = np.max([maxval, ticks[-1]])
            if xaxis:
                ax.set_xlim(minval, maxval)
            else:
                ax.set_ylim(minval, maxval)
            numticks = len(ticks)
            mul = int(np.ceil(numticks/6))
            ticks = ticks[::-mul][::-1]
            labels = labels[::-mul][::-1]
            if xaxis:
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)
            else:
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels)
        ax.grid(True, which = 'major')

    return locals
        
def test_log_ticks():
    test = np.arange(0.0001, 999) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(test, np.sqrt(test))
    set_log_ticks(ax, min(test), max(test), True)
    set_log_ticks(ax, min(np.sqrt(test)), max(np.sqrt(test)))
    fig.savefig('custom_ticks_demo.pdf')

def get_bcs(basedim, elasticity, spe = False, reference = False, ldomain = False):
    markers = list(range(1, 2*basedim+1))
    if ldomain:
        markers += list(range(7, 7+basedim))
    if not spe:
        const = 5e-1
        if elasticity:
            left_c = Constant([-const]+[0.]*(basedim-1))
            right_c = Constant([const]+[0.]*(basedim-1))
            left_q = Expression(['cc*(1.-x[1]*x[1])']+['0']*(basedim-1), cc = const, degree = 2)
            right_q = Expression(['cc*(x[1]*x[1]-1)']+['0']*(basedim-1), cc = const, degree = 2)
            bottom_c = Constant([0.]+[const]+[0.]*(basedim-2))
            top_c = Constant([0.]+[-const]+[0.]*(basedim-2))
            bottom_q = Expression(['0']+['cc*(1.-x[0]*x[0])']+['0']*(basedim-2), cc = const, degree = 2)
            top_q = Expression(['0']+['cc*(x[0]*x[0]-1)']+['0']*(basedim-2), cc = const, degree = 2)
            zero = Constant([0.]*basedim)
            fenics = Expression(['cc*x[0]*(1-x[1])']+['cc*x[1]*(1-x[0])']+['0']*(basedim-2), cc = const, degree = 3)
            pp = 0.3; ll = pp/((1.+pp)*(1-2*pp)); mm = 1./(2.*(1+pp))
            expr = Constant([const*(ll+mm)]+[const*(ll+mm)]+[0.]*(basedim-2))
        else:
            left_c = Constant(-const)
            right_c = Constant(const)
            left_q = Expression('cc*(1.-x[1]*x[1])', cc = const, degree = 2)
            right_q = Expression('cc*(x[1]*x[1]-1.)', cc = const, degree = 2)
            bottom_c = right_c
            top_c = left_c
            bottom_q = Expression('cc*(x[0]*x[0]-1.)', cc = const, degree = 2)
            top_q = Expression('cc*(1.-x[0]*x[0])', cc = const, degree = 2)
            zero = Constant(0.)
            if not ldomain:
                fenics = Expression('1.+0.25*(x[0]+1.)*(x[0]+1.)+0.5*(x[1]+1.)*(x[1]+1.)', degree = 2)
                expr = Constant(-3./2.)
            else:
                if basedim == 2:
                    fenics = Expression('[](double rr, double phi) { return pow(rr, 2./3.)*sin(2./3.*phi); }(sqrt(x[0]*x[0]+x[1]*x[1]), fmod(atan2(-x[0], x[1])+2.*atan2(0,-1), 2.*atan2(0,-1)))', degree = 5)
                    expr = Constant(0)
                elif basedim == 3:
                    lbda = 0.25
                    fenics = Expression('[](double rr) { return pow(rr, lbda); }(sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]))', lbda=lbda, degree = 5)
                    expr = Expression('[](double rr) { return -lbda*(ldbda+1)*pow(rr, lbda-2); }(sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])', lbda=lbda, degree = 3)
        if not reference:
            return [([(left_c, 1), (right_c, 2)], [(bottom_c, 3), (top_c, 4)], bottom_c), 
                    ([(left_q, 1), (right_q, 2)], [(bottom_q, 3), (top_q, 4)], None), 
                    ([(fenics, ii) for ii in markers], [], expr)]
        else:
            return [([(left_c, 1), (right_c, 2)], [], None), 
                    ([(left_c, 1), (right_c, 2)], [], bottom_c), 
                    ([(left_q, 1), (right_q, 2)], [], None), 
                    ([(left_q, 1), (right_q, 2)], [], bottom_c), 
                    ([(fenics, ii) for ii in markers], [], expr), 
                    ([(zero, ii) for ii in markers], [], expr), 
                    ([], [(bottom_c, 3), (top_c, 4)], None), 
                    ([], [(bottom_c, 3), (top_c, 4)], bottom_c), 
                    ([], [(bottom_q, 3), (top_q, 4)], None), 
                    ([], [(bottom_q, 3), (top_q, 4)], bottom_c)]
    else:
        assert(basedim == 3)
        in_well_in = 2e4
        in_well_out = 1e4
        out_well_in = 4e3

        if elasticity:
            in_well_in_c = Constant((0, 0, in_well_in))
            in_well_out_c = Constant((0, 0, in_well_out))
            out_well_in_c = Constant((0, 0, -out_well_in))
        else:
            in_well_in_c = Constant(in_well_in)
            in_well_out_c = Constant(-in_well_out)
            out_well_in_c = Constant(out_well_in)

        return [([(in_well_in_c, 101), (in_well_out_c, 102), (out_well_in_c, 104)], [], None)]


def build_nullspace(VV, elasticity = False):
    tmp = Function(VV)
    if elasticity:
        basedim = VV.mesh().geometry().dim()
        assert(basedim == 2 or basedim == 3), 'dimension [{:d}] nullspace not implemented'.format(basedim)
        nullspace_basis = [tmp.vector().copy() for ii in range(3 if basedim == 2 else 6)]
        #translational
        VV.sub(0).dofmap().set(nullspace_basis[0], 1.0);
        VV.sub(1).dofmap().set(nullspace_basis[1], 1.0);
        #rotational
        VV.sub(0).set_x(nullspace_basis[basedim], -1.0, 1);
        VV.sub(1).set_x(nullspace_basis[basedim],  1.0, 0);
        if basedim == 3:
            #dim3 translation
            VV.sub(2).dofmap().set(nullspace_basis[2], 1.0);
            #dim3 rotation
            VV.sub(0).set_x(nullspace_basis[4],  1.0, 2);
            VV.sub(2).set_x(nullspace_basis[4], -1.0, 0);
            VV.sub(2).set_x(nullspace_basis[5],  1.0, 1);
            VV.sub(1).set_x(nullspace_basis[5], -1.0, 2);
    else:
        nullspace_basis = [tmp.vector().copy()]
        nullspace_basis[0][:] = 1.
    for xx in nullspace_basis:
        xx.apply("insert")
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()
    return basis

def krylov_solve(AA, xx, bb, *args):
    solver = KrylovSolver(*args)
    return solver.solve(AA, xx, bb)

def failsafe_solve(AA, xx, bb, krylov_args = ('cg', 'hypre_amg'), solver_args = ('pastix')):
    try:
        solver = KrylovSolver(*krylov_args)
        it = solver.solve(AA, xx, bb)
    except:
        solve(AA, xx, bb, *solver_args)
        it = 0
    return it

def compare_mesh(mesha, meshb):
    if mesha.num_vertices() != meshb.num_vertices():
        return False
    if mesha.num_cells() != meshb.num_cells():
        return False
    if mesha.num_facets() != meshb.num_facets():
        return False
    return True

adapt_cpp_code = """
#include<pybind11/pybind11.h>
#include<dolfin/adaptivity/adapt.h>
#include<dolfin/mesh/Mesh.h>
#include<dolfin/mesh/MeshFunction.h>

namespace py = pybind11;

PYBIND11_MODULE(SIGNATURE, m) {
   m.def("adapt", (std::shared_ptr<dolfin::MeshFunction<std::size_t>> (*)(const dolfin::MeshFunction<std::size_t>&,
                   std::shared_ptr<const dolfin::Mesh>)) &dolfin::adapt,
         py::arg("mesh_function"), py::arg("adapted_mesh"));
}
"""
adapt = compile_cpp_code(adapt_cpp_code).adapt

if __name__ == '__main__':
    with LogWrapper('./test1.log') as logger:
        logger.info('Test 1')
    with LogWrapper('./test2.log') as logger:
        logger.info('Test 2')
