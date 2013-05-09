import numpy as np
from collections import deque, namedtuple
from itertools import izip
import tables
import pandas

from echolect.core import subsectime
from echolect.tools.tools import valargmax

__all__ = ['shift_ambig', 'target_points', 'event_list', 'Clustering']

def shift_ambig(shape, center_idx, scaling, ambig):
    # want (o[0]:o[0]+shape[0], o[1]:o[1]+shape[1]) of ambig
    o = tuple((l - (l + 1)//2) - c for l, c in zip(ambig.shape, center_idx))
    
    aslc = [slice(max(o[k], 0), min(o[k] + shape[k], ambig.shape[k])) for k in xrange(len(o))]
    sslc = [slice(aslc[k].start - o[k], aslc[k].stop - o[k]) for k in xrange(len(o))]
    
    shifted = np.zeros(shape, ambig.dtype)
    shifted[sslc] = scaling*ambig[aslc]

    return shifted

def target_points(snr, ambig, snrthresh, maxpoints=10):
    # snr is 2D: (delay)x(frequency), for single pulse
    # ambig is the square of the delay-frequency pulse autocorrelation function
    #  with 0 delay and 0 frequency in the middle of the matrix
    # (ifftshift brings 0 delay-frequency to index 0,0)
    modsnr = snr.copy()
    targets = []

    for k in xrange(maxpoints):
        # find maximum snr point and test vs threshold
        max_snr, max_snr_idx = valargmax(modsnr)
        max_snr_idx = np.unravel_index(max_snr_idx, modsnr.shape)
        if max_snr > snrthresh:
            targets.append(max_snr_idx)
            # assume point target, subtract out ambiguity
            modsnr -= shift_ambig(modsnr.shape, max_snr_idx, max_snr, ambig)
        else:
            break

    return targets

def event_list(filename):
    # read the clustered events from pytable located at filename and return an event list
    h5file = tables.openFile(filename, mode='r')
    signals = h5file.root.signals

    cluster_ids = set()
    for row in signals:
        cluster_ids.add(row['cluster'])
    cluster_ids = sorted(cluster_ids)

    e = []
    for cluster in cluster_ids:
        e.append(pandas.DataFrame(signals.readWhere('cluster == {0}'.format(cluster))))

    h5file.close()

    return e

class Clustering(object):
    # using a modified DBSCAN

    SignalDtype = np.dtype([('t', np.float64),
                            ('r', np.float64),
                            ('v', np.float64),
                            ('snr', np.float64),
                            ('rcs', np.float64),
                            ('pulse_num', np.uint32),
                            ('cluster', np.uint32),
                            ('core', np.bool_)])
    
    def __init__(self, filename, eps=0.5, min_samples=5, tscale=1, rscale=1,
                 vscale=1, writenoise=False, expectedsignals=50000):
        # setup pytables for saving point and cluster information
        self._h5file = tables.openFile(filename, mode='w')
        self._table = self._h5file.createTable('/', 'signals', Clustering.SignalDtype,
                                               filters=tables.Filters(complevel=6, complib='blosc'),
                                               expectedrows=expectedsignals)
        self._table.cols.cluster.createCSIndex()
        
        self.eps = eps
        self.min_samples = min_samples
        self.tscale = tscale
        self.rscale = rscale
        self.vscale = vscale
        self._tscale_sq = tscale**2
        self._rscale_sq = rscale**2
        self._vscale_sq = vscale**2
        self._neg_eps_tscale = -eps*tscale
        self.writenoise = writenoise

        self._visited = deque()
        self._visited_dists = deque()
        self._new = deque()
        self._next_cluster = 1

    def distance(self, p, o):
        dt = p['t'] - o['t']
        dr = p['r'] - o['r']
        dv = p['v'] - o['v']
        av = 0.5*(p['v'] + o['v'])

        dist = np.sqrt(dt**2/self._tscale_sq
                       + (dr - av*dt)**2/self._rscale_sq
                       + dv**2/self._vscale_sq)
        return dist

    def addnext(self, **kwargs):
        # add new signal point with values specified by keyword arguments
        p = np.zeros(1, Clustering.SignalDtype)
        for key, value in kwargs.iteritems():
            p[key] = value
        self._new.append(p)

        t_visit = p['t'][0] + self._neg_eps_tscale
        while self._new[0]['t'][0] < t_visit:
            # we have enough subsequent points to determine if next new point is a core point
            self._visitnew()

    def finish(self):
        # complete clustering when no new points will be added
        # cluster remaining new points (and write out remaining data for visited points)
        while len(self._new) > 0:
            self._visitnew()
        self._h5file.close()

    def _visitnew(self):
        # take the first point off the new deque and visit it
        p = self._new.popleft()
        
        # find neighbors from visited points
        neighbors = []
        for o, odists in izip(self._visited, self._visited_dists):
            # distances to p (first point in new_points deque) are the first
            # values in odists, pop them so this is true for new point next time
            if odists.popleft() < self.eps:
                neighbors.append(o)

        # calculate distances to other new points
        try:
            newarr = np.concatenate(self._new)
        except ValueError:
            newdists = deque()
        else:
            newdists = deque(self.distance(p, newarr))

        # find neighbors from new points
        for o, odist in izip(self._new, newdists):
            if odist < self.eps:
                neighbors.append(o)

        # cluster
        if len(neighbors) >= self.min_samples:
            # p is a core point, it belongs in a cluster
            p['core'] = True

            # determine/assign cluster for p
            pclust = p['cluster'][0]
            if pclust == 0:
                # create new cluster
                pclust = self._next_cluster
                self._next_cluster += 1
                p['cluster'] = pclust

            # find clusters of neighboring core points for merging
            neighb_clusts = set()
            noncore_neighb = []
            for o in neighbors:
                if o['core'][0]:
                    if o['cluster'][0] != pclust:
                        neighb_clusts.add(o['cluster'][0])
                else:
                    noncore_neighb.append(o)
            # merge clusters, switching neighbors' cluster to pclust
            for merge_clust in neighb_clusts:
                for on in self._new:
                    if on['cluster'][0] == merge_clust:
                        on['cluster'] = pclust
                for ov in self._visited:
                    if ov['cluster'][0] == merge_clust:
                        ov['cluster'] = pclust
                for orow in self._table.where('cluster == {0}'.format(merge_clust)):
                    orow['cluster'] = pclust
                    orow.update()
            self._table.flush()

            # p is core, so add all unclustered, non-core neighbors to same cluster
            for o in noncore_neighb:
                if o['cluster'][0] == 0:
                    o['cluster'] = pclust

        # add p to visited points and store its distances
        self._visited.append(p)
        self._visited_dists.append(newdists)

        # check and expire points that are too far away to be newly classified
        # (distance deque has run out of values)
        while len(self._visited_dists) > 0 and len(self._visited_dists[0]) == 0:
            self._expire_oldest_visited()

    def _expire_oldest_visited(self):
        # write data for oldest visited point to file and remove from memory
        self._visited_dists.popleft()
        o = self._visited.popleft()
        if (o['cluster'][0] != 0) or self.writenoise:
            self._table.append(o)
