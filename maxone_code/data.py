import h5py
import numpy as np
import parse
import pathlib
from cached_property import cached_property

from maxone_code.util import SIZE_CHANNEL,SAMPLING_FREQ,elecid2posi

class RawFile:
    def __init__(self, path):
        self.path = path
        self.h5f = h5py.File(path, 'r')
        self.spikes_maxone = self.h5f['proc0']['spikeTimes']

        self.lsb=self.h5f["settings"]["lsb"][0]
        self.lsb_uV=self.lsb*1.0E6

    def disp(self):
        print('sig shape:',self.sig.shape)
        print('record time: {}[s]'.format(self.sig.shape[1] // SAMPLING_FREQ))
        keys=self.elec2channel_table.keys()
        print('record eles_size: ', len(keys))
        print('elec ids', list(keys))
        print('channel ids', list(self.channel2elec_table.keys()))

    @property
    def sig(self):
        return self.h5f['sig']

    @cached_property
    def record_elecs(self):
        infos = self.h5f['mapping']
        elecs=[]
        for i, info in enumerate(infos):
            electro_id = info[1]
            elecs.append(electro_id)
        return elecs

    @cached_property
    def elec_posi(self):
        return list(map(lambda x: elecid2posi(x), self.record_elecs))

    @cached_property
    def channel2elec_table(self):
        table = {}
        infos = self.h5f['mapping']
        for i, info in enumerate(infos):
            c_id = info[0]
            e_id = info[1]
            table[c_id] = e_id
        return table

    def channel2elec(self,channel_id):
        return self.channel2elec_table[channel_id]

    @cached_property
    def elec2channel_table(self):
        table={}
        infos = self.h5f['mapping']
        for i, info in enumerate(infos):
            c_id = info[0]
            e_id = info[1]
            table[e_id] = c_id
        return table

    def elec2channel(self,elec_id):
        return self.elec2channel_table[elec_id]

    @cached_property
    def channelmap(self):
        # channel_id -> xdata_elec_id
        # -1 if unmapped, elec_id else
        infos = self.h5f['mapping']
        xdata_elec_id_table = np.zeros(SIZE_CHANNEL, dtype=int)
        xdata_elec_id_table.fill(-1)
        for i, info in enumerate(infos):
            channel_id = info[0]
            electro_id = info[1]
            xdata_elec_id = i
            xdata_elec_id_table[channel_id] = xdata_elec_id
        return xdata_elec_id_table

    @cached_property
    def stimsig(self):
        filename = self.path.name.split('.')[0] + '_cached_stimsig'
        savepath = self.path.parents[0].joinpath(filename)
        if savepath.exists():
            stimsig = np.load(savepath)
            return stimsig
        else:
            sig = self.sig
            length = sig.shape[1]
            stimsig = np.zeros(length)
            sig.read_direct(stimsig, source_sel=np.s_[SIZE_CHANNEL, :length], dest_sel=np.s_[:length])
            np.save(savepath, stimsig)
            return stimsig

    def get_sig_from_channel(self,channel,start,stop):
        length=stop-start
        sig=np.zeros(length)
        sig.read_direct(self.sig, source_sel=np.s_[channel, start:stop], dest_sel=np.s_[:length])
        return sig


class CfgFile:
    def __init__(self, path):

        file = open(path, "r")
        source = file.readline()

        all_elec_id = []
        for i in parse.findall("({:d})", source):
            all_elec_id.append(i.fixed[0])
        all_elec_id = np.array(all_elec_id)
        self.record_elec_id = all_elec_id

        # channel_id -> elec_id: if not connected, value is -1
        chan_elec_table = np.full(SIZE_CHANNEL,-1)
        for i,j in parse.findall("{:d}({:d})", source):
            chan_elec_table[i] = j
        self.chan_elec_table = chan_elec_table


class Experiment:
    def __init__(self,path,interval=None):
        self.rawfile = RawFile(path)
        if interval is None:
            self.INTERVAL = 20000
        else:
            self.INTERVAL=interval

    def disp(self):
        self.rawfile.disp()
        print('trial start frame: ',self.trial_start_frames)
        print('stim interval: ',self.intervals)

    @cached_property
    def trial_start_frames(self):
        stim_frames = list(np.where(self.rawfile.stimsig != 512)[0])
        trial_start_frames = []
        trial_start_frames.append(stim_frames[0])
        for i in stim_frames:
            interval = i - trial_start_frames[-1]
            if interval < self.INTERVAL:
                continue
            else:
                trial_start_frames.append(i)
        return  trial_start_frames

    @cached_property
    def intervals(self):
        a = np.array(self.trial_start_frames)
        b = a[1:]
        a = a[:-1]
        return b-a
