import numpy as np

from maxone_code.data import Experiment

from maxone_code.fig.multiunit import FigMultiUnit
from maxone_code.util import distElecId
from maxone_code.spikedetect import BandPassFilter
from cached_property import cached_property
from sklearn.decomposition import PCA


class ArtifactAssay:

    def __init__(self, path, lookforframe=None):
        self.path=path
        self.exp = Experiment(path)

        self.STIM_ELEC = int(self.path.name.split('.')[0].split('-')[-1])
        self.STIM_CHANNEL = self.exp.rawfile.elec2channel(self.STIM_ELEC)

        self.SIZE_TRIAL = 5
        self.SIZE_CHANNEL = 24

        if lookforframe is None:
            self.SIZE_FRAME = 100  # for 5ms
        else:
            self.SIZE_FRAME = lookforframe

        self.SATURATION_FOCUS_FRAMES = 40  # range to looking for end frame of saturation
        self.SATU_VAL = 1023

        self.drawer=FigMultiUnit()
        self.assay_frame_range=None

    def exec(self):
        self.loaddata()
        self.butterpass()
        self.cleanArtifact()

    def disp(self):
        self.disp_raw(isConcat=True)
        self.disp_filtered(isConcat=True)
        self.disp_artifacts()
        self.disp_cleaned(isConcat=True)

    def loaddata(self):
        forcus_channels = list(self.exp.rawfile.channel2elec_table.keys())
        forcus_channels.remove(self.STIM_CHANNEL)
        forcus_channels.sort(key=lambda x:x)
        def sortKey(chan):
            elec1=self.STIM_ELEC
            elec2=self.exp.rawfile.channel2elec(chan)
            return distElecId(elec1,elec2)
        forcus_channels.sort(key=sortKey)
        self.forcus_channels=forcus_channels

        assert (self.SIZE_TRIAL == len(self.exp.trial_start_frames))
        assert (self.SIZE_CHANNEL == len(forcus_channels))

        channel2data_table = {}
        data2channel_table = {}
        for i, chan in enumerate(forcus_channels):
            channel2data_table[chan] = i
            data2channel_table[i] = (chan,sortKey(chan))
        self.channel2data_table=channel2data_table
        self.data2channel_table=data2channel_table

        data_raw = np.zeros((self.SIZE_TRIAL, self.SIZE_CHANNEL, self.SIZE_FRAME))
        source = self.exp.rawfile.sig
        for i, start_frame in enumerate(self.exp.trial_start_frames):
            for j, channel in enumerate(forcus_channels):
                s_slice = np.s_[channel, start_frame:start_frame + self.SIZE_FRAME]
                d_slice = np.s_[:self.SIZE_FRAME]
                source.read_direct(data_raw[i][j], source_sel=s_slice, dest_sel=d_slice)
        self.data_raw=data_raw

    @cached_property
    def saturation_emd_frames(self):
        end_frames=[]
        for data in self.data_raw:
            frames = np.where(data[:][:self.SATURATION_FOCUS_FRAMES] == self.SATU_VAL)[1]
            end_frames.append(np.max(frames))
        return end_frames

    def disp_raw(self, isAll=False, isConcat=False):
        if isAll is True:
            data=self.data_raw
        else:
            if isConcat is True:
                size_ch=self.data_raw.shape[1]  # chan
                data = [self.data_raw.transpose(1,0,2).reshape(size_ch,-1)]
            else:
                data=[self.data_raw[0]]
        for i,trial in enumerate(data):
            self.drawer.drawWithSatuLine(trial,self.saturation_emd_frames[i],ylabel='[bit]',title='sig data')

    def butterpass(self):
        bpf=BandPassFilter()
        self.data_filtered= bpf.filter(self.data_raw)

    def disp_filtered(self, isAll=False, isConcat=False, yrange=None):
        if self.assay_frame_range is None:
            data=self.data_filtered[:,:,:]
        else:
            data=self.data_filtered[:,:,self.assay_frame_range]

        if isAll is True:
            data=data
        else:
            if isConcat is True:
                size_ch=data.shape[1]  # chan
                data = [data.transpose(1,0,2).reshape(size_ch,-1)]
            else:
                data=[data[0]]
        for i,trial in enumerate(data):
            self.drawer.drawWithSatuLine(trial,self.saturation_emd_frames[i],ylabel='[bit]',draw_xaxis=True,title='post bandpass filter',yrange=yrange)

    def disp_relatioin(self, isAll=False):
        if isAll is True:
            data=self.data_filtered.copy()
        else:
            data=[self.data_filtered[0].copy()]

        xs=[]
        for i, v in enumerate(self.forcus_channels):
            chan, dist = self.data2channel_table[i]
            assert (v==chan)
            xs.append(dist)


        for i,trial in enumerate(data):
            relation=np.transpose(trial) # (time,chan)
            for series in relation:
                series /= max(series.min(), series.max(), key=abs)

            yss=relation[:self.saturation_emd_frames[i]]
            self.drawer.drawRelation(xs,yss)

            yss=relation[self.saturation_emd_frames[i]+1:self.saturation_emd_frames[i]+6]
            self.drawer.drawRelation(xs,yss)
            yss=relation[self.saturation_emd_frames[i]+6:self.saturation_emd_frames[i]+11]
            self.drawer.drawRelation(xs,yss)
            yss=relation[self.saturation_emd_frames[i]+11:self.saturation_emd_frames[i]+16]
            self.drawer.drawRelation(xs,yss)
            # self.drawer.drawRelation(xs,relation[:])

    def cleanArtifact(self,slice=None):
        self.assay_frame_range=slice

        # prepare chan_dist_table
        size_ch=len(self.forcus_channels)
        chan_dist_table=np.zeros((self.SIZE_CHANNEL,self.SIZE_CHANNEL))
        for i,chan_i in enumerate(self.forcus_channels):
            for j,chan_j in enumerate(self.forcus_channels):
                if j >= i: continue
                elec1 = self.exp.rawfile.channel2elec(chan_i)
                elec2 = self.exp.rawfile.channel2elec(chan_j)
                chan_dist_table[i,j] = distElecId(elec1, elec2, is_uM=True)
                chan_dist_table[j,i] = distElecId(elec1, elec2, is_uM=True)
        self.chan_dist_table=chan_dist_table

        # pca
        data_filtered=self.data_filtered.copy()
        if slice is not None:
            data_filtered=data_filtered[:,:,slice]
        data = data_filtered.transpose((0, 2, 1))  # (trial,chan,time) -> (trial,time,chan)
        data = data.reshape(-1, data.shape[-1])  # -> (trial*time,chan)
        pca = PCA(n_components=5)
        pca.fit(data)

        # inferred artifact
        self.org_components=pca.components_.copy()
        artifacts = np.zeros((data.shape))
        for i, chan_i in enumerate(self.forcus_channels):
            components = self.org_components.copy()
            for j, chan_j in enumerate(self.forcus_channels):
                if self.chan_dist_table[i, j] < 20:
                    components[:, j] = 0
            for c in components:
                c /= np.linalg.norm(c)

            pca.components_ = components
            artifact_t = pca.transform(data)
            artifact = np.matmul(artifact_t, self.org_components)
            artifacts[:, i] = artifact[:, i]
        self.artifacts = artifacts.transpose() # (trial*time,chan) -> (chan,trial*time)

        artifacts=self.artifacts.reshape(self.SIZE_CHANNEL,self.SIZE_TRIAL,-1)  # (chan,trial,time)
        artifacts=artifacts.transpose((1,0,2))  # (trail,chan,time)

        self.data_cleaned = data_filtered - artifacts

    def disp_artifacts(self):
        self.drawer.draw(self.artifacts,ylabel='[bit]',draw_xaxis=True,title='Artifact')

    def disp_cleaned(self, isAll=False, isConcat=False, yrange=None):
        if isAll is True:
            data=self.data_cleaned
        else:
            if isConcat is True:
                size_ch=self.data_cleaned.shape[1]  # chan
                data = [self.data_cleaned.transpose(1,0,2).reshape(size_ch,-1)]
            else:
                data=[self.data_cleaned[0]]
        for i,trial in enumerate(data):
            self.drawer.draw(trial,ylabel='[bit]',draw_xaxis=True,title='post Artifact removal',yrange=yrange)
