import os, sys, getopt, re
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from paraview.simple import *
import numpy as np
import vtk.numpy_interface.dataset_adapter as dsa

def usage():
    print('''Usage
python3 plotenrichments.py
Option          Default                 Info
-h --help                               help
--basename=                             path to xdmf of enrichments
--number=       5                       number of enrichments to plot''')

if __name__ == '__main__':
#   enrichments_path = '/home/wu/atacama_data/homogenization/enrichments_inside_refined'
#   mesh_path = '/home/wu/atacama_data/homogenization/fem_inside_refined'
#   fig_path = '/home/wu/atacama_data/enrichments_figs/2d'

    enrichments_path = '/data/wu/homogenization/enrichments_inside_refined'
    mesh_path = '/data/wu/homogenization/fem_inside_refined'
    fig_path = '/data/wu/enrichments_figs/2d'


    basename = 'square_channel/res_5/beta_5.00e+00/contrast_1.00e+04/heat/0/patch_4/neumann/coeffs/square_channel_5_0_patch_4_shapes_liptonE.xdmf'
    number = 4

    try:
        opts, args = getopt.getopt(sys.argv[1:],'h',['help','basename=','number='])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h','--help'): 
            usage()
            sys.exit()
        elif opt == '--basename':
            basename = arg if len(arg) > 5 else basename
        elif opt == '--number':
            number = int(arg) if int(arg) > 0 else number

    fill = int(np.log(number)/np.log(10.))+1

    mm = re.search('(heat|elasticity)', basename)
    if mm.group(0) == 'heat':
        mm = re.search('(?<=heat/)\d+', basename)
    else:
        mm = re.search('(?<=elasticity/)\d+', basename)
    patch_level = mm.group(0)

    mm = re.search('patch_\d+', basename)
    patch = mm.group(0)

    mm = re.search('.*patch_\d+/\w+', basename)
    basefig = mm.group(0)
    mm = re.search('(?<=shapes_)\w+', basename)
    basefig = '{:s}/{:s}/{:s}'.format(fig_path, basefig, mm.group(0))

    mm = re.search('^\w+', basename)
    inctype = mm.group(0)
    
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # get active layout
    layout1 = GetLayout()

    # patch outline
    print('Loading patch facets')
    patch_4_facetsxdmf = XDMFReader(FileNames=['{:s}/{:s}/{:s}_5_patches/{:s}/{:s}_facets.xdmf'.format(mesh_path, inctype, inctype,  patch_level, patch)])
    patch_4_facetsxdmf.CellArrayStatus = ['facet_function']
    patch_4_facetsxdmf.GridStatus = ['facets']

    clip1 = Clip(Input=patch_4_facetsxdmf)
    clip1.Scalars = ['CELLS', 'facet_function']
    clip1.ClipType = 'Scalar'
    clip1.Value = 1.0

    calculator1 = Calculator(Input=clip1)
    calculator1.ResultArrayName = 'black'
    calculator1.Function = '1'

    blackLUT = GetColorTransferFunction('black')

    calculator1Display = Show(calculator1, renderView1)
    calculator1Display.ColorArrayName = ['POINTS', 'black']
    calculator1Display.LookupTable = blackLUT
    calculator1Display.LineWidth = 3.0

    blackLUT.ApplyPreset('Grayscale', True)
    blackLUT.RGBPoints = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    # read coeffs
    print('Loading coefficients')
    square_channel_5_0_patch_4_shapes_liptonExdmf = XDMFReader(FileNames=['{:s}/{:s}'.format(enrichments_path, basename)])
    square_channel_5_0_patch_4_shapes_liptonExdmf.PointArrayStatus = ['u']

    # get animation scene
    animationScene1 = GetAnimationScene()
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # color scheme for u
    uLUT = GetColorTransferFunction('u')
#   uLUT.ColorSpace = 'RGB'
#   uLUT.RGBPoints = [1.0, 0.0, 0.0, 1.0,
#                     2.0, 0.4, 0.4, 0.9,
#                     3.0, 0.8, 0.8, 0.8,
#                     4.0, 0.9, 0.4, 0.4,
#                     5.0, 1.0, 0.0, 0.0]
#   uLUT.UseLogScale = 0
    uLUT.Discretize = 0
    uLUT.LockDataRange = 1
    uLUT.RescaleOnVisibilityChange = 0
    uLUT.UseLogScale = 0
    uLUT.ApplyPreset('BuRd', True)
    uPWF = GetOpacityTransferFunction('u')

    uLUTColorBar = GetScalarBar(uLUT, renderView1)
    uLUTColorBar.DrawAnnotations = 0
    uLUTColorBar.LabelFormat = '%.1e'
    uLUTColorBar.NumberOfLabels = 5
    uLUTColorBar.DrawTickMarks = 1
    uLUTColorBar.DrawTickLabels = 1
    uLUTColorBar.RangeLabelFormat = '%.1e'
    uLUTColorBar.TitleOpacity = 1.0
    uLUTColorBar.TitleFontFamily = 'Courier'
    uLUTColorBar.TitleBold = 1
    uLUTColorBar.TitleFontSize = 12
    uLUTColorBar.LabelOpacity = 1.0
    uLUTColorBar.LabelFontFamily = 'Courier'
    uLUTColorBar.LabelBold = 1
    uLUTColorBar.LabelFontSize = 12

    # compute gradients
    computeDerivatives1 = ComputeDerivatives(Input=square_channel_5_0_patch_4_shapes_liptonExdmf)
    computeDerivatives1.Scalars = ['POINTS', 'u']
    computeDerivatives1.Vectors = [None, '']

    calculator2 = Calculator(Input=computeDerivatives1)
    calculator2.AttributeMode = 'Cell Data'
    calculator2.ResultArrayName = 'norm(grad(u))'
    calculator2.Function = 'mag(ScalarGradient)'

    # color scheme for norm(grad(u))
    normgraduLUT = GetColorTransferFunction('normgradu')
#   normgraduLUT.ColorSpace = 'RGB'
#   normgraduLUT.RGBPoints = [1e0, 0.0, 0.0, 1.0,
#                             1e1, 0.4, 0.4, 0.9,
#                             1e2, 0.8, 0.8, 0.8,
#                             1e3, 0.9, 0.4, 0.4,
#                             1e4, 1.0, 0.0, 0.0]
#   normgraduLUT.UseLogScale = 1
    normgraduLUT.Discretize = 0
    normgraduLUT.LockDataRange = 1
    normgraduLUT.RescaleOnVisibilityChange = 0
    normgraduLUT.UseLogScale = 1
    normgraduLUT.MapControlPointsToLogSpace()
    normgraduLUT.ApplyPreset('BuRd', True)
    normgraduPWF = GetOpacityTransferFunction('normgradu')

    normgraduLUTColorBar = GetScalarBar(normgraduLUT, renderView1)
    normgraduLUTColorBar.Title = '|grad(u)|'
    normgraduLUTColorBar.ComponentTitle = ''
    normgraduLUTColorBar.DrawAnnotations = 0
    normgraduLUTColorBar.LabelFormat = '%.1e'
    normgraduLUTColorBar.NumberOfLabels = 5
    normgraduLUTColorBar.DrawTickMarks = 1
    normgraduLUTColorBar.DrawTickLabels = 1
    normgraduLUTColorBar.RangeLabelFormat = '%.1e'
    normgraduLUTColorBar.TitleOpacity = 1.0
    normgraduLUTColorBar.TitleFontFamily = 'Courier'
    normgraduLUTColorBar.TitleBold = 1
    normgraduLUTColorBar.TitleFontSize = 12
    normgraduLUTColorBar.LabelOpacity = 1.0
    normgraduLUTColorBar.LabelFontFamily = 'Courier'
    normgraduLUTColorBar.LabelBold = 1
    normgraduLUTColorBar.LabelFontSize = 12

    # warp by u
    warpByScalar1 = WarpByScalar(Input=calculator2)
    warpByScalar1.Scalars = ['POINTS', 'u']
    warpByScalar1Display = Show(warpByScalar1, renderView1)

    # u contour
    contour1 = Contour(Input=warpByScalar1)
    contour1.ContourBy = ['POINTS', 'u']
    contour1.ComputeNormals = 0
    contour1.ComputeScalars = 1
    contour1.GenerateTriangles = 0
    contour1.Isosurfaces = np.linspace(-1.995,1.995,400)

    contour1Display = Show(contour1, renderView1)
    ColorBy(contour1Display, ('POINTS', 'u'))
    contour1Display.LineWidth = 3.0
    contour1Display.LookupTable = uLUT

    # norm(grad(u)) to point data
    cellDatatoPointData1 = CellDatatoPointData(Input=warpByScalar1)

    # norm(grad(u)) contour
    contour2 = Contour(Input=cellDatatoPointData1)
    contour2.ContourBy = ['POINTS', 'norm(grad(u))']
    contour2.ComputeNormals = 0
    contour2.ComputeScalars = 1
    contour2.GenerateTriangles = 0
    contour2.Isosurfaces = 10.**np.linspace(-4,2,61)

    contour2Display = Show(contour2, renderView1)
    ColorBy(contour2Display, ('POINTS', 'norm(grad(u))'))
    contour2Display.LineWidth = 3.0
    contour2Display.LookupTable = normgraduLUT

    os.makedirs(basefig, exist_ok=True)

    # flat 2d plots
    warpByScalar1.ScaleFactor = 0.0
    renderView1.InteractionMode = '2D'
    renderView1.ViewSize = [2500, 2000]
    renderView1.CameraPosition = [0.0, 0.0, 2.42343008666464]
    renderView1.CameraFocalPoint = [0.0, 0.0, 0.020812280476093292]
    renderView1.CameraViewUp = [0., 1., 0.]
    renderView1.CameraParallelScale = 0.47459807846155255
    renderView1.ResetCamera()

    animationScene1.GoToFirst()
    for ii in range(number):
        suffix = str(ii).zfill(fill)
        rawData = servermanager.Fetch(cellDatatoPointData1)
        data = dsa.WrapDataObject(rawData)
        umin = np.min(data.PointData['u'])
        umax = np.max([np.max(data.PointData['u']), umin+np.abs(umin)*0.0001])
        gradmin = np.max([np.min(data.PointData['norm(grad(u))']), 1e-4])
        gradmax = np.max([np.max(data.PointData['norm(grad(u))']), gradmin*1.0001])

        # plot u
        ColorBy(warpByScalar1Display, ('POINTS', 'u'))
        Hide(contour2, renderView1)
        HideScalarBarIfNotNeeded(normgraduLUT, renderView1)
        warpByScalar1Display.LookupTable = uLUT
        warpByScalar1Display.SetScalarBarVisibility(renderView1, True)
        Show(contour1, renderView1)
        uLUT.RescaleTransferFunction(umin, umax)
        uPWF.RescaleTransferFunction(umin, umax)
        SaveScreenshot('{:s}/u_{:s}.png'.format(basefig, suffix), layout=layout1, magnification=1, quality=100)

        # plot norm(grad(u))
        ColorBy(warpByScalar1Display, ('CELLS', 'norm(grad(u))'))
        Hide(contour1, renderView1)
        HideScalarBarIfNotNeeded(uLUT, renderView1)
        warpByScalar1Display.LookupTable = normgraduLUT
        warpByScalar1Display.SetScalarBarVisibility(renderView1, True)
        Show(contour2, renderView1)
        normgraduLUT.RescaleTransferFunction(gradmin, gradmax)
        normgraduPWF.RescaleTransferFunction(gradmin, gradmax)
        SaveScreenshot('{:s}/grad(u)_{:s}.png'.format(basefig, suffix), layout=layout1, magnification=1, quality=100)
        
        animationScene1.GoToNext()

    # 3d plots
        
    renderView1.InteractionMode = '3D'
    renderView1.ViewSize = [3000, 2000]
    renderView1.CameraPosition = [-1.04917297288668, -1.81844063415343, 0.340293115912126]
    renderView1.CameraFocalPoint = [0.100303333580533, 0.172510730744394, -0.0650742571184371]
    renderView1.CameraViewUp = [0.08682408883346521, 0.15038373318043502, 0.9848077530122081]
    renderView1.ResetCamera()

    animationScene1.GoToFirst()
    for ii in range(number):
        suffix = str(ii).zfill(fill)
        rawData = servermanager.Fetch(cellDatatoPointData1)
        data = dsa.WrapDataObject(rawData)
        umin = np.min(data.PointData['u'])
        umax = np.max([np.max(data.PointData['u']), umin+np.abs(umin)*0.0001])
        warpByScalar1.ScaleFactor = 0.3/np.max([np.abs(umin),np.abs(umax)])
        gradmin = np.max([np.min(data.PointData['norm(grad(u))']), 1e-4])
        gradmax = np.max([np.max(data.PointData['norm(grad(u))']), gradmin*1.0001])

        # plot u
        ColorBy(warpByScalar1Display, ('POINTS', 'u'))
        Hide(contour2, renderView1)
        HideScalarBarIfNotNeeded(normgraduLUT, renderView1)
        warpByScalar1Display.LookupTable = uLUT
        warpByScalar1Display.SetScalarBarVisibility(renderView1, True)
        Show(contour1, renderView1)
        uLUT.RescaleTransferFunction(umin, umax)
        uPWF.RescaleTransferFunction(umin, umax)
        SaveScreenshot('{:s}/displaced_u_{:s}.png'.format(basefig, suffix), layout=layout1, magnification=1, quality=100)

        # plot norm(grad(u))
        ColorBy(warpByScalar1Display, ('CELLS', 'norm(grad(u))'))
        Hide(contour1, renderView1)
        HideScalarBarIfNotNeeded(uLUT, renderView1)
        warpByScalar1Display.LookupTable = normgraduLUT
        warpByScalar1Display.SetScalarBarVisibility(renderView1, True)
        Show(contour2, renderView1)
        normgraduLUT.RescaleTransferFunction(gradmin, gradmax)
        normgraduPWF.RescaleTransferFunction(gradmin, gradmax)
        SaveScreenshot('{:s}/displaced_grad(u)_{:s}.png'.format(basefig, suffix), layout=layout1, magnification=1, quality=100)

        animationScene1.GoToNext()

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
