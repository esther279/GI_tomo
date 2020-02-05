#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Test 
from skimage.filters import gaussian
from skimage.segmentation import active_contour

img = recon
s = np.linspace(0, 2*np.pi, 50)
r = 17 + 10*np.sin(s)
c = 34 + 10*np.cos(s)
init = np.array([r, c]).T

'''
image : (N, M) or (N, M, 3) ndarray
    Input image.
snake : (N, 2) ndarray
    Initial snake coordinates. For periodic boundary conditions, endpoints
    must not be duplicated.
alpha : float, optional
    Snake length shape parameter. Higher values makes snake contract
    faster.
beta : float, optional
    Snake smoothness shape parameter. Higher values makes snake smoother.
w_line : float, optional
    Controls attraction to brightness. Use negative values to attract toward
    dark regions.
w_edge : float, optional
    Controls attraction to edges. Use negative values to repel snake from
    edges.
gamma : float, optional
    Explicit time stepping parameter.
bc : {'periodic', 'free', 'fixed'}, optional
    Boundary conditions for worm. 'periodic' attaches the two ends of the
    snake, 'fixed' holds the end-points in place, and 'free' allows free
    movement of the ends. 'fixed' and 'free' can be combined by parsing
    'fixed-free', 'free-fixed'. Parsing 'fixed-fixed' or 'free-free'
    yields same behaviour as 'fixed' and 'free', respectively.
max_px_move : float, optional
    Maximum pixel distance to move per iteration.
max_iterations : int, optional
    Maximum iterations to optimize snake shape.
convergence: float, optional
    Convergence criteria.
    '''

snake = active_contour(gaussian(img, 1),
                       init, alpha=0.06, beta=3, bc='fixed', w_line=0.1, w_edge=0.5)

plt.figure(401, figsize=[20,10]); plt.clf()
plt.imshow(img, cmap=plt.cm.gray)
plt.plot(init[:, 1], init[:, 0], '--r', lw=3)
plt.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
plt.axis([0, img.shape[1], img.shape[0], 0])

plt.show()


import skimage.draw as draw
import skimage.segmentation as seg


image_labels = np.zeros(img.shape, dtype=np.uint8)
indices = draw.circle_perimeter(25, 25, 24)
image_labels[indices] = 1

r = 37 + 1*np.sin(s)
c = 18 + 1*np.cos(s)
init = np.array([r, c]).T
image_labels[init[:, 1].astype(np.int), init[:, 0].astype(np.int)] = 2

r = 15 + 0.1*np.sin(s)
c = 20 + 0.1*np.cos(s)
init = np.array([r, c]).T
image_labels[init[:, 1].astype(np.int), init[:, 0].astype(np.int)] = 2

plt.figure(401, figsize=[20,10]); plt.clf()
plt.subplot(211)
plt.imshow(image_labels);
plt.imshow(img, alpha=0.3);
plt.axis([0, img.shape[1], img.shape[0], 0])

image_segmented = seg.random_walker(img, image_labels)
plt.subplot(212)
plt.imshow(image_segmented);
plt.axis([0, img.shape[1], img.shape[0], 0])











