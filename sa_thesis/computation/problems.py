import dolfin
import numpy

from ..helpers.utils import solve

class Definitions:
	def __init__(self, mesh, *, domains = None, facets = None):
		self.mesh = mesh
		self.coefficient_space = dolfin.FunctionSpace(mesh, 'DG', 0)
		self.basedim = mesh.geometric_dimension()
		if domains is None:
			self.domains = MeshFunction('size_t', mesh, self.basedim, 0)
		else:
			self.domains = domains
		self.domains_array = self.domains.array()
		self.domains_values = numpy.unique(domains.array())
		self.facets = facets
	
	def where(self, domain):
		return numpy.where(self.domains_array == domain)[0]
	
	def coefficient_function(self):
		return dolfin.Function(self.coefficient_space)
	
	def assemble(self, ff, gs = [], hs = [], *, mesh = None, facets = None):
		if mesh is None:
			mesh = self.mesh
		space = self.getSpace(mesh)
		if facets is None:
			facets = self.facets
		dx = dolfin.Measure('dx', domain = mesh)
		ds = dolfin.ds if facets is None else dolfin.Measure('ds', domain = mesh, subdomain_data = facets)
		test = dolfin.TestFunction(space)
		trial = dolfin.TrialFunction(space)
		kk = dolfin.inner(self.sigma(trial), self.epsilon(test))*dx
		ll = dolfin.inner(ff, test)*dx
		for hh, domain in hs:
			ll += dolfin.inner(hh, test)*ds(domain)
		bcs = []
		for gg, domain in gs:
			bcs.append(dolfin.DirichletBC(space, gg, facets, domain))
		AA, bb = dolfin.assemble_system(kk, ll, bcs)
		return AA, bb
		
	def assemble_and_solve(self, ff, gs = [], hs = [], *, mesh = None, facets = None):
		if mesh is None:
			mesh = self.mesh
		AA, bb = self.assemble(ff, gs, hs, mesh = mesh, facets = facets)
		uu = dolfin.Function(self.getSpace(mesh))
		solve(AA, uu, bb)
		return uu

	def default_coefficients(self, *, coefficient = None):
		if coefficient is None:
			coefficient = self.default_coefficient()
		return [coefficient]*len(self.domains_values)

	def coefficient_functions(self, coefficients = None):
		if coefficients is None:
			coefficients = self.default_coefficients()
		functions = {}
		for key in coefficients[0].keys():
			functions[key] = self.coefficient_function()
		for domain in self.domains_values:
			where = self.where(domain)
			coefficient = coefficients[domain]
			for key in coefficient.keys():
				functions[key].vector()[where] = coefficient[key]
		return functions

class ScalarDefinitions(Definitions):
	def __init__(self, mesh, *, domains = None, facets = None):
		super().__init__(mesh, domains = domains, facets = facets)
		self.getSpace = lambda mesh = mesh: dolfin.FunctionSpace(mesh, 'CG', 1)

class VectorDefinitions(Definitions):
	def __init__(self, mesh, *, domains = None, facets = None):
		super().__init__(mesh, domains = domains, facets = facets)
		self.getSpace = lambda mesh = mesh: dolfin.VectorFunctionSpace(mesh, 'CG', 1)

class PoissonProblem(ScalarDefinitions):
	def __init__(self, mesh, *, domains = None, facets = None, coefficients = None):
		super().__init__(mesh, domains = domains, facets = facets)
		coeffs = self.coefficient_functions(coefficients)
		self.kappa = coeffs['aa']

	def epsilon(self, uu):
		return dolfin.grad(uu)

	def sigma(self, uu):
		return self.kappa*dolfin.grad(uu)

	@classmethod
	def default_coefficient(cls, *, aa = 1.):
		return {'aa': aa}

class LinearElasticityProblem(VectorDefinitions):
	def __init__(self, mesh, *, domains = None, facets = None, coefficients = None):
		super().__init__(mesh, domains = domains, facets = facets)
		coeffs = self.coefficient_functions(coefficients)
		EE = coeffs['EE']; nu = coeffs['nu']
		self.ll = (nu*EE) / ((1.+nu)*(1.-2.*nu))
		self.mu = EE / (2.*(1.+nu))

	def epsilon(self, uu):
		return dolfin.sym(dolfin.grad(uu))

	def sigma(self, uu):
		return 2.*self.mu*self.epsilon(uu) + self.ll*dolfin.tr(self.epsilon(uu))*dolfin.Identity(self.basedim)

	@classmethod
	def default_coefficient(cls, *, EE = 1e6, nu = 0.25):
		return {'EE': EE, 'nu': nu}

class OrthotropicMaterialProblem(VectorDefinitions):
	def __init__(self, mesh, *, domains = None, facets = None, coefficients = None):
		super().__init__(mesh, domains = domains, facets = facets)
		coeffs = self.coefficient_functions(coefficients)
		cos = dolfin.cos(coeffs['angle'])
		sin = dolfin.sin(coeffs['angle'])
		ax_0, ax_1, ax_2 = coeffs['axis_0'], coeffs['axis_1'], coeffs['axis_2'] 
		E_0, E_1, E_2 = coeffs['E_0'], coeffs['E_1'], coeffs['E_2']
		nu_01, nu_02, nu_12 = coeffs['nu_01'], coeffs['nu_02'], coeffs['nu_12']
		self.G_01, self.G_02, self.G_12 = coeffs['G_01'], coeffs['G_02'], coeffs['G_12']
		self.RR = dolfin.as_matrix([
			[cos + ax_0*ax_0*(1 - cos), ax_0*ax_1*(1 - cos) - ax_2*sin, ax_0*ax_2*(1 - cos) + ax_1*sin],
			[ax_0*ax_1*(1 - cos) + ax_2*sin, cos + ax_1*ax_1*(1 - cos), ax_1*ax_2*(1 - cos) - ax_0*sin],
			[ax_0*ax_2*(1 - cos) - ax_1*sin, ax_1*ax_2*(1 - cos) + ax_0*sin, cos + ax_2*ax_2*(1 - cos)]
		])
		nu_10 = nu_01*E_1/E_0; nu_20 = nu_02*E_2/E_0; nu_21 = nu_12*E_2/E_1
		kk = 1 - nu_01*nu_10 - nu_12*nu_21 - nu_02*nu_20 - 2*nu_01*nu_12*nu_20
		self.DD = dolfin.as_matrix([
			[E_0*(1 - nu_12*nu_21)/kk, E_0*(nu_12*nu_20 + nu_10)/kk, E_0*(nu_21*nu_10 + nu_20)/kk],
			[E_1*(nu_02*nu_21 + nu_01)/kk, E_1*(1 - nu_02*nu_20)/kk, E_1*(nu_20*nu_01 + nu_21)/kk],
			[E_2*(nu_01*nu_12 + nu_02)/kk, E_2*(nu_10*nu_02 + nu_12)/kk, E_2*(1 - nu_01*nu_10)/kk]
		])	 

	def epsilon(self, uu):
		return dolfin.sym(dolfin.grad(uu))

	def sigma(self, uu):
		eps = self.RR*self.epsilon(uu)*self.RR.T
		ss = self.DD*dolfin.as_vector([eps[0, 0], eps[1, 1], eps[2, 2]])
		ss_01 = 2*self.G_01*eps[0, 1]
		ss_02 = 2*self.G_02*eps[0, 2]
		ss_12 = 2*self.G_12*eps[1, 2]
		return self.RR.T*dolfin.as_matrix([[ss[0], ss_01, ss_02], [ss_01, ss[1], ss_12], [ss_02, ss_12, ss[2]]])*self.RR

	@classmethod
	def default_coefficient(cls, *, axis = numpy.array([0, 0, 1.]), angle = 0.,
		E_0 = 1.5e7, E_1 = 1e6, E_2 = 1e6, nu_01 = 0.3, nu_02 = 0.3, nu_12 = 0.44, G_01 = 7.5e5, G_02 = 7.5e5, G_12 = 1e6/(2*(1+0.44))
	):
		return {
			'axis_0': axis[0], 'axis_1': axis[1], 'axis_2': axis[2], 'angle': angle,
			'E_0': E_0, 'E_1': E_1, 'E_2': E_2, 'nu_01': nu_01, 'nu_02': nu_02, 'nu_12': nu_12, 'G_01': G_01, 'G_02': G_02, 'G_12': G_12
		}