import matplotlib.pyplot as plt
import numpy as np
import itertools

def wireSphere(ax):
    # draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = .3*np.outer(np.cos(u), np.sin(v))
    y = .3*np.outer(np.sin(u), np.sin(v))
    z = .3*np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color="k", rstride=5, cstride=5, linewidth=.5)

def visualizeWeights(W_unit):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    wireSphere(ax)
    W_samples = W_unit.flatten(0, 6).squeeze().detach().numpy()
    ax.scatter(W_samples[:, 0], W_samples[:, 1], W_samples[:, 2], s=1)
    plt.show()


def sphereSlice(ax):
    q = 1  # defines upper starting point of the spherical segment
    p = 6/9  # defines ending point of the spherical segment as ratio

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(q, p * np.pi, int(p * 100))
    x = .2 * np.outer(np.cos(u), np.sin(v))
    y = .2 * np.outer(np.sin(u), np.sin(v))
    z = .2  * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='k', rcount=30, ccount=30, linewidth=10)


def visualizeViews(x, y, z, c, ax, lim=1):

    #wireSphere(ax)
    #ax.scatter(0, 0, 0, s=1000, c='k', marker=(5,2), linewidths=3)
    ax.scatter(x, y, z, s=50, c=c, marker=9)
    ax.set_xlim(lim, -lim)
    ax.set_ylim(lim, -lim)
    ax.set_zlim(lim, -lim)


def sph2cart(a, e, d):
    """ converts azimuth a, elevation e, and distance d to cartesian coordinates x y z"""

    phi = np.radians(90 - e)
    theta = np.radians(a)
    rho = d

    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)

    return x,y,z


def splitSet(data, criterion):
    train_set, test_set = list(), list()
    for d in data:
        (test_set, train_set)[d in criterion(d)].append(d)

    return train_set, test_set


if __name__ == '__main__':



    view_dist = {"azimuth": {"range": (0, 360),
                             "views": 18},
                 "elevation": {"range": (-30, 30),
                               "views": 6},
                 "dist": {"range": (1, 2.5),
                          "views": 3}}

    azimuths = np.linspace(view_dist['azimuth']['range'][0], view_dist['azimuth']['range'][1],
                           num=view_dist['azimuth']['views'], endpoint=False).tolist()
    elevations = np.linspace(view_dist['elevation']['range'][0], view_dist['elevation']['range'][1],
                             num=view_dist['elevation']['views']).tolist()
    dists = np.linspace(view_dist['dist']['range'][0], view_dist['dist']['range'][1],
                        num=view_dist['dist']['views']).tolist()


    # List Criterions here -------- IMPLEMENT THESE IN DATALOADERS!
    criterion_elevation_stepover = lambda a: (a[1] == -30 or a[1] == -6 or a[1] == 18)
    criterion_elevation_darkside = lambda a: (a[1] < -0)

    criterion_azimuth_darkside = lambda a: (a[0] < 180)
    criterion_azimuth_stepover = lambda a: (a[0] % 40 == 0)
    criterion_azimuth_steplarger = lambda a: (a[0] % 60 == 0)

    criterion_distance_stepover = lambda a: (a[2] == 1.75)

    ea = list(itertools.product(azimuths, elevations, dists))
    train_set, test_set = list(), list()
    for sph in ea:
        (test_set, train_set)[criterion_azimuth_steplarger(sph)].append(sph)
        print("train size:{}\t test size:{}".format(len(train_set), len(test_set)) )


    # random.shuffle(ea)
    # mid_ind = len(list(ea))//2
    # train_set = ea[:mid_ind]
    # test_set = ea[mid_ind:]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    c = 'r'
    xs = list()
    ys = list()
    zs = list()
    for s in test_set:
        x, y, z = sph2cart(s[0], s[1], s[2])

        xs.append(x)
        ys.append(y)
        zs.append(z)

    visualizeViews(xs, ys, zs, c, ax, lim=2.5)

    c = 'b'
    xs = list()
    ys = list()
    zs = list()
    for s in train_set:
        x, y, z = sph2cart(s[0], s[1], s[2])

        xs.append(x)
        ys.append(y)
        zs.append(z)

    visualizeViews(xs, ys, zs, c, ax, lim=2.5)
    plt.show()
