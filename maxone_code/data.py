import h5py
import numpy as np
import parse
import pathlib

from maxone_code.util import SIZE_CHANNEL,SAMPLING_FREQ

class RawFile:
    def __init__(self, path):
        self.path = path
        self.h5f = h5py.File(path, 'r')
        self.spikes_maxone = self.h5f['proc0']['spikeTimes']

        # channel_id -> xdata_elec_id
        # 0 if unmapped, elec_id else
        infos = self.h5f['mapping']
        xdata_elec_id_table = np.zeros(SIZE_CHANNEL, dtype=int)
        xdata_elec_id_table.fill(-1)
        for i, info in enumerate(infos):
            channel_id = info[0]
            electro_id = info[1]
            xdata_elec_id = i
            xdata_elec_id_table[channel_id] = xdata_elec_id
        self.channelmap = xdata_elec_id_table

        self.lsb=self.h5f["settings"]["lsb"][0]
        self.lsb_uV=self.lsb*1.0E6

        self._stimsig=None

    @property
    def sig(self):
        return self.h5f['sig']

    @property
    def stimsig(self):
        if self._stimsig is not None:
            return self._stimsig
        else:
            path = pathlib.Path(self.path).parent.joinpath('stimsig.npy')
            if path.exists():
                _stimsig = np.load(str(path))
                self._stimsig = _stimsig
                return _stimsig
            else:
                sig = self.sig
                length = sig.shape[1]
                _stimsig = np.zeros(length)
                sig.read_direct(_stimsig, source_sel=np.s_[SIZE_CHANNEL, :length], dest_sel=np.s_[:length])
                np.save(str(path), _stimsig)
                self._stimsig = _stimsig
                return _stimsig

    def disp(self):
        print('sig shape:',self.sig.shape)
        print('record time: {}[s]'.format(self.sig.shape[1] // SAMPLING_FREQ))
        print('record channels:' ,len(list(filter(lambda x:x>0,self.channelmap))) )

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
