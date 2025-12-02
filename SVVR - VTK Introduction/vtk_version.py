#!/usr/bin/env vtkpython

import vtk

v = vtk.vtkVersion()
version = v.GetVTKSourceVersion()
print(version)
