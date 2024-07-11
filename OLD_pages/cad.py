import pyvista as pv
import streamlit as st
from stpyvista import stpyvista

# ipythreejs does not support scalar bars :(
pv.global_theme.show_scalar_bar = False

st.title("A cube")
st.info("""Code adapted from https://docs.pyvista.org/user-guide/jupyter/pythreejs.html#scalars-support""")

## Initialize a plotter object
plotter = pv.Plotter(window_size=[400,400])

kinds = [
    'tetrahedron',
    'cube',
    'octahedron',
    'dodecahedron',
    'icosahedron',
]
centers = [
    (0, 1, 0),
    (0, 0, 0),
    (0, 2, 0),
    (-1, 0, 0),
    (-1, 2, 0),
]

solids = [pv.PlatonicSolid(kind, radius=0.4, center=center) for kind, center in zip(kinds, centers)]

for mesh in solids:
    ## Add mesh to the plotter
    plotter.add_mesh(mesh, cmap='bwr', line_width=1)

## Final touches
plotter.view_isometric()
plotter.background_color = 'white'

## Send to streamlit
stpyvista(plotter, key="pv_cube")