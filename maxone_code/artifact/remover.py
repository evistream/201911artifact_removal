import numpy as np
import pathlib
from cached_property import cached_property
from sklearn.decomposition import PCA
from scipy import interpolate

from maxone_code.data import Experiment
from maxone_code.fig.multiunit import FigMultiUnit
from maxone_code.fig.mea import FigMea
from maxone_code.util import distElecId
from maxone_code.spikedetect import BandPassFilter,HighPassFilter
import maxone_code.util as util



# - assay.data_raw   :　sigデータ
# - assay.data_filtered   :バンドパスフィルター後
# - (assay.data_resampled)  : リサンプリング後
# - assay.artifacts_c   :チャネル間で共通するアーチファクト
# - assay.data_cleaned_c   :チャネル間アーチファクトの除去後
# - assay.artifacts_t   :トライアル間で共通するアーチファクト
# - assay.data_cleaned_t   :トライアル間アーチファクトの除去後
# - assay.data_all_cleaned   :バンドパスフィルター後

class ArtifactAssay(object):

    # 電極・チャネルは以下のidで表現される。それらのうち一つが決まれば、すべて一意に決まる。
    # elec_id : 電極を表すid
    # chan_id : チャンネルを表すid
    # xid : 解析用データ列(data_*)上での列数または行数を表すid

    def __init__(self, path, lookforframe=None):
        self.path = path
        self.exp = Experiment(path)

        self.STIM_ELEC = int(self.path.name.split('.')[0].split('-')[-1])
        self.STIM_CHANNEL = self.exp.rawfile.elec2channel(self.STIM_ELEC)

        self.SIZE_TRIAL = 5

        if lookforframe is None:
            self.SIZE_FRAME = 100  # for 5ms
        else:
            self.SIZE_FRAME = lookforframe

        self.BPF_MARGIN_FRAME = 300
        self.MARGIN_FRAME_FOR_SPLINE = self.BPF_MARGIN_FRAME + 30
        self.Kn = 4
        self.ELIMINATE_DISTANCE = 200

        self.SATURATION_FOCUS_FRAMES = 40  # range to looking for end frame of saturation
        self.SATU_VAL = 1023

        self.SPLINE_K = 3
        self.RESAMPLE_RATE =  10

        self.drawer = FigMultiUnit()
        self.meadrawer = FigMea()

        self.bpf = BandPassFilter(lowcut=250, highcut=3000, order=4)
        # self.bpf = HighPassFilter(lowcut=250,order=4)

        self.assay_frame_range = None

        forcus_channels = list(self.exp.rawfile.channel2elec_table.keys())
        forcus_channels.remove(self.STIM_CHANNEL)
        # forcus_channelsを刺激電極から近い順に並び替える
        # 距離が同じ場合は、さらに若い番号順に並び替え
        def sortKey(chan):
            elec1 = self.STIM_ELEC
            elec2 = self.exp.rawfile.channel2elec(chan)
            dist = distElecId(elec1, elec2, is_uM=True)
            return (dist, elec2)
        forcus_channels.sort(key=sortKey)
        self.forcus_channels = forcus_channels
        assert (self.SIZE_TRIAL == len(self.exp.trial_start_frames))
        # assert (self.SIZE_CHANNEL == len(forcus_channels))
        self.SIZE_CHANNEL = len(self.forcus_channels)
        self.SIZE_MAX_CHANNEL = util.SIZE_CHANNEL # Maxoneの最大計測数

        channel2xid_table = {}
        xid2channel_table = {}
        for i, chan in enumerate(self.forcus_channels):
            channel2xid_table[chan] = i
            dist_from_stim = sortKey(chan)[0]
            xid2channel_table[i] = (chan, dist_from_stim)
        self.channel2xid_table = channel2xid_table
        self.xid2channel_table = xid2channel_table

    def disp(self):
        self.disp_raw()
        self.disp_filtered()
        self.disp_artifacts_c()
        self.disp_cleaned_c()
        self.disp_artifacts_t()
        self.disp_cleaned_t()
        self.disp_allcleaned()

    def filterProccessing(self, filter_obj, raw_data, isContainMargin=True):
        if isContainMargin:
            return filter_obj.filter(raw_data)
        else:
            MARGIN = self.BPF_MARGIN_FRAME
            return filter_obj.filter(raw_data)[:, :, MARGIN:-MARGIN]

    def loaddata(self, cache_path=None):
        MARGIN = self.MARGIN_FRAME_FOR_SPLINE

        is_cache_exist = cache_path is not None and pathlib.Path(cache_path).exists()
        if is_cache_exist :
            self.data_raw_m_for_spline = np.load(cache_path)
        else:
            ## this block take 5min when lookupframe=1000, trials=5, channels=500
            LENGTH = self.SIZE_FRAME + 2 * MARGIN
            data_pre_butter = np.zeros((self.SIZE_TRIAL, self.SIZE_CHANNEL, LENGTH))
            source = self.exp.rawfile.sig
            channels = list(self.channel2xid_table.keys())
            for i, stim_frame in enumerate(self.exp.trial_start_frames):
                start_frame = stim_frame - MARGIN
                assert (start_frame > 0)

                ## complicated
                SIZE_SIG_X = source.shape[0]
                data = np.zeros((SIZE_SIG_X, LENGTH))
                s_slice = np.s_[:, start_frame:start_frame + LENGTH]
                d_slice = np.s_[:, :LENGTH]
                source.read_direct(data, source_sel=s_slice, dest_sel=d_slice)
                data_pre_butter[i] = data[channels]
                # simpler code is following, but slower. it may take 1.5 time longer.
                # for j, channel in enumerate(self.forcus_channels):
                #     s_slice = np.s_[channel, start_frame:start_frame + self.SIZE_FRAME]
                #     d_slice = np.s_[:self.SIZE_FRAME]
                #     source.read_direct(data_pre_butteer[i][j], source_sel=s_slice, dest_sel=d_slice)

            self.data_raw_m_for_spline = data_pre_butter
            if cache_path is not None:
                self.cache_data_raw_m_for_spline(cache_path)

        MARGIN_ = MARGIN - self.BPF_MARGIN_FRAME
        self.data_raw_m = self.data_raw_m_for_spline[:, :, MARGIN_:-MARGIN_]  # 接尾語'_m'はマージン付きの意味
        self.data_raw = self.data_raw_m_for_spline[:, :, MARGIN:-MARGIN]

        data_post_butter = self.filterProccessing(self.bpf,self.data_raw_m_for_spline)
        self.data_filtered_m_for_spline = data_post_butter
        self.data_filtered_m = data_post_butter[:, :, MARGIN_:-MARGIN_]
        self.data_filtered = data_post_butter[:, :, MARGIN:-MARGIN]

    def cache_data_raw_m_for_spline(self,path):
        np.save(path, self.data_raw_m_for_spline)

    def make_resample(self):
        '''
        アーチファクトの時間遅れを考慮し、スプライン補間で位置を調整するプログラム。
        3次でスプライン補間をして、10倍の解像度にする。
        アーチファクトの立ち下がりとx軸の交点から、アーチファクトの時間方向の位置を検出する。
        その交点を中心に元の周波数でリサンプリングする。
        '''
        k = self.SPLINE_K
        RESAMPLE_RATE = self.RESAMPLE_RATE
        margin_frames = self.MARGIN_FRAME_FOR_SPLINE*RESAMPLE_RATE
        look_for_frame = np.s_[margin_frames+30:margin_frames+90]
        frames_from_start_to_center = 6

        shape = self.data_filtered_m_for_spline.shape
        shape_ = (shape[2] * RESAMPLE_RATE,)
        splined_data = np.zeros(shape_)  # (frame_m_for_spline)
        data_resampled_m = np.zeros(self.data_filtered_m.shape)  # (trial, chan, frame_m)

        no_triggered_chan = set()  # あるtrialでx軸との交差点が見つからなかった電極

        for i, trial in enumerate(self.data_filtered_m_for_spline):
            for j, chan in enumerate(trial):
                x = np.arange(0, shape[2], 1)
                s = interpolate.InterpolatedUnivariateSpline(x, chan, k=k)
                xnew = np.arange(0, shape[2], 1 / RESAMPLE_RATE)

                splined_data = s(xnew)

                start = look_for_frame.start
                stop = look_for_frame.stop

                a = splined_data[start:stop - 1]
                b = splined_data[start + 1:stop]
                c = b * a
                indices = np.array(np.where(c < 0)).T
                if len(indices) == 0:
                    data_resampled_m[i, j, :] = 0
                    no_triggered_chan.add(j)
                else:
                    center = indices[0, 0] + look_for_frame.start
                    start = center - (self.BPF_MARGIN_FRAME + frames_from_start_to_center) * RESAMPLE_RATE
                    stop = start + self.data_filtered_m.shape[2] * RESAMPLE_RATE
                    data_resampled_m[i, j, :] = splined_data[start:stop:RESAMPLE_RATE]

        self.data_resampled_m = data_resampled_m
        self.data_resampled = data_resampled_m[:, :, self.BPF_MARGIN_FRAME:-self.BPF_MARGIN_FRAME]
        self.spline_no_triggered_chan = no_triggered_chan

    @cached_property
    def record_eles(self):
        record_eles = {}
        for name, value in self.xid2channel_table.items():
            chan, _ = value
            posi = self.exp.rawfile.channel2posi(chan)
            record_eles[name] = posi
        return record_eles

    @cached_property
    def stim_eles(self):
        posi = self.exp.rawfile.elec2posi(self.STIM_ELEC)
        stim_eles = {
            'st': posi
        }
        return stim_eles

    @cached_property
    def mea_xid_table(self):
        onset = np.array(list(self.record_eles.values())).min(axis=0)
        offset = np.array(list(self.record_eles.values())).max(axis=0)
        subplot_size = offset - onset + [1, 1]
        subplot_table = np.zeros(subplot_size, dtype=int)
        subplot_table.fill(-1)
        for k, posi in self.record_eles.items():
            index = np.array(list(posi)) - onset
            index = tuple(index)
            subplot_table[index] = k

        # set stim_xposi
        posi = self.exp.rawfile.elec2posi(self.STIM_ELEC)
        self._stim_xposi = posi - onset

        return subplot_table

    @cached_property
    def stim_xposi(self):
        _ = self.mea_xid_table
        return self._stim_xposi

    @cached_property
    def saturation_end_frames(self):
        end_frames = np.zeros(self.data_raw.shape[0])
        for i, data in enumerate(self.data_raw):
            try:
                frames = np.where(data[:][:self.SATURATION_FOCUS_FRAMES] == self.SATU_VAL)[1]
                end_frames[i] = np.max(frames)
            except:
                pass
        return end_frames

    @cached_property
    def chan_dist_table(self):
        size_ch = len(self.forcus_channels)
        chan_dist_table = np.zeros((self.SIZE_CHANNEL, self.SIZE_CHANNEL))
        for i, chan_i in enumerate(self.forcus_channels):
            for j, chan_j in enumerate(self.forcus_channels):
                if j >= i: continue
                elec1 = self.exp.rawfile.channel2elec(chan_i)
                elec2 = self.exp.rawfile.channel2elec(chan_j)
                chan_dist_table[i, j] = distElecId(elec1, elec2, is_uM=True)
                chan_dist_table[j, i] = distElecId(elec1, elec2, is_uM=True)
        return chan_dist_table

    @cached_property
    def chan_stim_dist_array(self):
        chan_stim_dist_array = np.zeros(len(self.forcus_channels))
        for i, c in enumerate(self.forcus_channels):
            elec1 = self.STIM_ELEC
            elec2 = self.exp.rawfile.channel2elec(c)
            dist = distElecId(elec1, elec2, is_uM=True)
            chan_stim_dist_array[i] = dist
        return chan_stim_dist_array

    def cleanArtifact(self, slice=None, eliminate_dist=None, ignore_first_trial=False, isResample=False):
        # eliminate_dist [uM] :== int
        self.assay_frame_range = slice
        if eliminate_dist is None:
            eliminate_dist = self.ELIMINATE_DISTANCE

        # data for pca ::= (trial,chan,time)
        # data_m for transform, which includes BPF_MARGIN_FRAME
        # when highpassfilter, BPF_MARGIN_FRAME needs
        def exclude_margin(data):
            return data[:,:,self.BPF_MARGIN_FRAME:-self.BPF_MARGIN_FRAME]

        # pca
        if isResample is False:
            data_pca = self.data_filtered.copy()
        else:
            data_pca = self.data_resampled.copy()
        if slice is not None:
            data_pca = data_pca[:, :, slice]
        if ignore_first_trial is True:
            data_pca = data_pca[1:, :, :]
        shape = data_pca.shape  # (trial,chan,time)
        data_pca = data_pca.transpose((0, 2, 1))  # (trial,chan,time) -> (trial,time,chan)
        data_pca = data_pca.reshape(-1, shape[1])  # -> (trial*time,chan)
        pca = PCA(n_components=self.Kn)
        pca.fit(data_pca)
        scores = pca.transform(data_pca) # -> (trial*time, Kn)
        scores = scores.reshape(-1, shape[2], self.Kn)  # (trial*time,Kn) -> (trial,time,Kn)
        self.pca_scores_c = scores.transpose(0, 2, 1)  # (trial,time,Kn) -> (trial,Kn,time)

        # inferred artifact
        if isResample is False:
            data_m = self.data_filtered_m.copy()
        else:
            data_m = self.data_resampled_m.copy()
        data_m = data_m.transpose((0, 2, 1))  # (trial,chan,time) -> (trial,time,chan)
        data_m = data_m.reshape(-1, data_m.shape[-1])  # -> (trial*time,chan)
        self.org_components = pca.components_.copy()
        artifacts_m = np.zeros((data_m.shape))
        for i, chan_i in enumerate(self.forcus_channels):
            components = self.org_components.copy()
            for j, chan_j in enumerate(self.forcus_channels):
                if self.chan_dist_table[i, j] < eliminate_dist:
                    components[:, j] = 0
            for c in components:
                c /= np.linalg.norm(c)

            pca.components_ = components
            artifact_t_m = pca.transform(data_m)
            b = np.mean(data_m, axis=0)
            artifact_m = np.matmul(artifact_t_m, self.org_components) + b
            artifacts_m[:, i] = artifact_m[:, i]
        artifacts_m = artifacts_m.reshape(self.SIZE_TRIAL, -1, self.SIZE_CHANNEL)  # (trial*time,chan) -> (trial,time,chan)
        artifacts_m = artifacts_m.transpose(0, 2, 1)  # (trial,time,chan) -> (trial,chan,time)

        self.artifacts_c_m = artifacts_m
        self.artifacts_c = exclude_margin(self.artifacts_c_m)

        if isResample is False:
            self.data_cleaned_c_m = self.data_filtered_m - artifacts_m  # (trial,chan,time)
        else:
            self.data_cleaned_c_m = self.data_resampled_m - artifacts_m  # (trial,chan,time)
        self.data_cleaned_c_m = self.data_filtered_m - artifacts_m  # (trial,chan,time)
        self.data_cleaned_c = exclude_margin(self.data_cleaned_c_m)

        data_pca = self.data_cleaned_c.copy()
        if slice is not None:
            data_pca = data_pca[:, :, slice]
        data_pca = data_pca.transpose((1, 2, 0))  # (trial,chan,time) -> (chan,time,trial)
        data_m = self.data_cleaned_c_m.copy()
        data_m = data_m.transpose((1, 2, 0))  # (trial,chan,time) -> (chan,time,trial)
        artifacts_m = np.zeros((data_m.shape))

        for i, chan_i in enumerate(self.forcus_channels):
            # pca
            pca = PCA(n_components=self.Kn)
            pca.fit(data_pca[i])
            self.org_components = pca.components_.copy()

            for j in range(data_m.shape[2]):  # trial size
                components = self.org_components.copy()
                components[:, j] = 0
                for c in components:
                    c /= np.linalg.norm(c)

                pca.components_ = components
                artifact_t_m = pca.transform(data_m[i])
                artifact_m = np.matmul(artifact_t_m, self.org_components)
                artifacts_m[i, :, j] = artifact_m[:, j]
        artifacts_m = artifacts_m.transpose((2, 0, 1))  # (chan,time,trial) -> (trial,chan,time)

        self.artifacts_t_m = artifacts_m
        self.artifacts_t = exclude_margin(self.artifacts_t_m)

        self.artifacts_all_m = self.artifacts_c_m + self.artifacts_t_m
        self.artifacts_all = exclude_margin(self.artifacts_all_m)

        self.data_cleaned_t_m = self.data_cleaned_c_m - artifacts_m
        self.data_cleaned_t = exclude_margin(self.data_cleaned_t_m)

        self.data_all_cleaned_m = self.bpf.filter(self.data_cleaned_t_m)
        self.data_all_cleaned = exclude_margin(self.data_all_cleaned_m)


    def filter_artifacts_with_dist(self, artifacts, drange):
        data = artifacts.copy()
        if drange is None:
            return data
        else:
            chan_ids = (self.chan_stim_dist_array > drange[0]) * (self.chan_stim_dist_array < drange[1])
            if chan_ids.any():
                data = data[chan_ids, :]
                return data
            else:
                print('no data for range', drange)
                return None

    def filter_data_with_chan(self, data, drange, xids):
        data = data.copy()
        if drange is None:
            if xids is None:
                return data
            else:
                if isinstance(xids, int):
                    xids = [xids]
                return data[:, xids]
        else:
            chan_ids = (self.chan_stim_dist_array > drange[0]) * (self.chan_stim_dist_array < drange[1])
            if chan_ids.any():
                data = data[:, chan_ids, :]
                return data
            else:
                print('no data for range', drange)
                return None


    def disp_data(self, data, isConcat=True, ylabel=None, draw_xaxis=False, title=None, trange=None, yrange=None,
                  drange=None, frange=None, legend=False, xids=None, path=None, isFrameScale=True):
        # data ::= (trial, channel, frame)
        data = data.copy()

        # channel範囲
        # xidで指定
        data = self.filter_data_with_chan(data, drange, xids)

        # trial範囲
        if trange is not None:
            if isinstance(trange, int):
                data=data[[trange]]
            else:
                data=data[trange]

        # frame範囲
        if frange is not None:
            data = data[:, :, frange]
        else:
            if self.assay_frame_range is not None:
                data = data[:, :, self.assay_frame_range]

        # trialで結合させるかどうか
        if isConcat is True:
            size_ch = data.shape[1]  # chan
            data = data.transpose(1, 0, 2).reshape(1, size_ch, -1)

        for i, trial in enumerate(data):
            self.drawer.draw(trial, ylabel=ylabel, draw_xaxis=draw_xaxis, title=title, yrange=yrange,
                             legend=legend, path=None, isFrameScale=isFrameScale)

    def disp_artifacts_c(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','Artifact')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.artifacts_c, **kwargs)

    def disp_artifacts_t(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','Artifact')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.artifacts_t, **kwargs)

    def disp_artifacts_all(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','Artifact')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.artifacts_all, **kwargs)

    def disp_raw(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','sig data')
        self.disp_data(self.data_raw, **kwargs)

    def disp_filtered(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','post bandpass filter')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.data_filtered, **kwargs)

    def disp_resampled(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','post bandpass filter resampled')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.data_resampled, **kwargs)

    def disp_cleaned_c(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','post Artifact removal over channel')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.data_cleaned_c, **kwargs)

    def disp_cleaned_t(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','post all Artifact removal')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.data_cleaned_t, **kwargs)

    def disp_allcleaned(self, **kwargs):
        kwargs.setdefault('ylabel','[bit]')
        kwargs.setdefault('title','post removal, bandpass filter')
        kwargs.setdefault('draw_xaxis',True)
        self.disp_data(self.data_all_cleaned, **kwargs)

    def disp_mea(self, **kwargs):
        self.meadrawer.draw_all(self.record_eles, self.stim_eles, **kwargs)

    def disp_mea_local(self, **kwargs):
        kwargs.setdefault('range','auto')
        self.meadrawer.draw_local(self.record_eles, self.stim_eles, **kwargs)

    def subplot_template(self, table, center, stride, subplot_size=None, parity=None):
        # table ::= [[int]]
        # center ::= [int, int]
        # stride ::= int
        # parity ::= 0 | 1 | None. if 1, the new table include stim_elec
        table=table.copy()
        offset = center % stride
        new_center = center // stride
        new_table = table[offset[0]::stride,offset[1]::stride]

        if parity is not None:
            center_parity = np.sum(new_center) % 2
            if (center_parity + parity) % 2 == 0:
                new_table[::2,::2] = -1
                new_table[1::2,1::2] = -1
            else:
                new_table[::2,1::2] = -1
                new_table[1::2,::2] = -1

        if subplot_size is not None:
            sp_offset_x = max(new_center[0] - subplot_size[0] // 2, 0)
            sp_offset_y = max(new_center[1] - subplot_size[1] // 2, 0)
            new_table = new_table[sp_offset_x:sp_offset_x + subplot_size[0] ,
                        sp_offset_y:sp_offset_y + subplot_size[1] ]
        return new_table

    def disp_data_on_mea(self, data,**kwargs):
        kwargs.setdefault('share_axis', False)
        kwargs.setdefault('show_spine', True)
        sp_pattern=kwargs.pop('sp_pattern', None)  # ::= None | int
        if sp_pattern is not None:
            if isinstance(sp_pattern,int):
                subplot_patterns = [
                    (4,(10,10),0),
                    (2,(9,9),0),
                    (1,(5,5),None),
                ]
                sp_pattern=subplot_patterns[sp_pattern]
            subplot_table = self.subplot_template(self.mea_xid_table, self.stim_xposi,*sp_pattern)
        else:
            subplot_table = self.mea_xid_table
        self.drawer.draw_on_mea(data, subplot_table, **kwargs)

    def disp_data_on_mea_2(self, data1, data2,**kwargs):
        kwargs.setdefault('share_axis', False)
        kwargs.setdefault('show_spine', True)
        sp_pattern=kwargs.pop('sp_pattern', None)  # ::= None | int
        if sp_pattern is not None:
            if isinstance(sp_pattern,int):
                subplot_patterns = [  # (stride, subplot_size, parity)
                    (4,(10,10),0),
                    (2,(9,9),0),
                    (1,(5,5),None),
                ]
                sp_pattern=subplot_patterns[sp_pattern]
            subplot_table = self.subplot_template(self.mea_xid_table, self.stim_xposi,*sp_pattern)
        else:
            subplot_table = self.mea_xid_table
        colors=kwargs.pop('colors', None)  # ::= None | int
        self.drawer.draw_on_mea_2(data1, data2, subplot_table, colors=colors, **kwargs)

    def disp_artifacts_c_on_mea(self, **kwargs):
        kwargs.setdefault('title','Artifact')
        self.disp_data_on_mea(self.artifacts_c, **kwargs)

    def disp_artifacts_t_on_mea(self, **kwargs):
        kwargs.setdefault('title','Artifact')
        self.disp_data_on_mea(self.artifacts_t, **kwargs)

    def disp_artifacts_all_on_mea(self, **kwargs):
        kwargs.setdefault('title','Artifact')
        self.disp_data_on_mea(self.artifacts_all, **kwargs)

    def disp_raw_on_mea(self, **kwargs):
        kwargs.setdefault('title','sig data')
        self.disp_data_on_mea(self.data_raw, **kwargs)

    def disp_filtered_on_mea(self, **kwargs):
        kwargs.setdefault('title','post bandpass filter')
        self.disp_data_on_mea(self.data_filtered, **kwargs)

    def disp_cleaned_c_on_mea(self, **kwargs):
        kwargs.setdefault('title','post Artifact removal over channel')
        self.disp_data_on_mea(self.data_cleaned_c, **kwargs)

    def disp_cleaned_t_on_mea(self, **kwargs):
        kwargs.setdefault('title','post all Artifact removal')
        self.disp_data_on_mea(self.data_cleaned_t, **kwargs)

    def disp_allcleaned_on_mea(self, **kwargs):
        kwargs.setdefault('title','post removal and bandpass filter')
        self.disp_data_on_mea(self.data_all_cleaned, **kwargs)

    def disp_removal_c(self, **kwargs):
        colors=['black','red']
        kwargs.setdefault('colors',colors)
        self.disp_data_on_mea_2(self.data_filtered, self.artifacts_c, **kwargs)

    def disp_removal_t(self, **kwargs):
        colors=['black','red']
        kwargs.setdefault('colors',colors)
        self.disp_data_on_mea_2(self.data_cleaned_c, self.artifacts_t, **kwargs)

    def disp_beforeafter_c(self, **kwargs):
        colors=['black','blue']
        kwargs.setdefault('colors',colors)
        self.disp_data_on_mea_2(self.data_filtered, self.data_cleaned_c, **kwargs)

    def disp_beforeafter_t(self, **kwargs):
        colors=['black','blue']
        kwargs.setdefault('colors',colors)
        self.disp_data_on_mea_2(self.data_cleaned_c, self.data_cleaned_t, **kwargs)

    def disp_beforeafter_ct(self, **kwargs):
        colors=['black','blue']
        kwargs.setdefault('colors',colors)
        self.disp_data_on_mea_2(self.data_filtered, self.data_cleaned_c, **kwargs)

    def disp_beforeafter_all(self, **kwargs):
        colors=['black','blue']
        kwargs.setdefault('colors',colors)
        self.disp_data_on_mea_2(self.data_filtered, self.data_all_cleaned, **kwargs)



    #
    # def disp_relatioin(self, isAll=False):
    #     if isAll is True:
    #         data = self.data_filtered.copy()
    #     else:
    #         data = [self.data_filtered[0].copy()]
    #
    #     xs = []
    #     for i, v in enumerate(self.forcus_channels):
    #         chan, dist = self.xid2channel_table[i]
    #         assert (v == chan)
    #         xs.append(dist)
    #
    #     for i, trial in enumerate(data):
    #         relation = np.transpose(trial)  # (time,chan)
    #         for series in relation:
    #             series /= max(series.min(), series.max(), key=abs)
    #
    #         yss = relation[:self.saturation_end_frames[i]]
    #         self.drawer.drawRelation(xs, yss)
    #
    #         yss = relation[self.saturation_end_frames[i] + 1:self.saturation_end_frames[i] + 6]
    #         self.drawer.drawRelation(xs, yss)
    #         yss = relation[self.saturation_end_frames[i] + 6:self.saturation_end_frames[i] + 11]
    #         self.drawer.drawRelation(xs, yss)
    #         yss = relation[self.saturation_end_frames[i] + 11:self.saturation_end_frames[i] + 16]
    #         self.drawer.drawRelation(xs, yss)
    #         # self.drawer.drawRelation(xs,relation[:])