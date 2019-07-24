import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage.filters import maximum_filter1d
import os
import msgpack

import camera_models
import calibrate

the_grid = np.arange(7*5).reshape(5, 7)[::-1]

#print(the_grid)
#use_targets = [
#                  9,  10, 11,
#                  16, 17, 18,
#                  23, 24, 25,
#              ]
use_targets = [
        15,16,17,18,19,
         8, 9,10,11,12,
         1, 2, 3, 4, 6
        ]
#threshold = np.deg2rad(2.0)
threshold = 35
screen_size = 1920, 1080
quality_threshold = 0.5
camera = camera_models.load_intrinsics(".", "Pupil Cam1 ID2", screen_size)


def load_session(pupil_log, time_delta=0):
    #info = {k: v for k, v in (r.split(',',1) for r in open(info))}
    #time_delta = float(info["Start Time (System)"]) - float(info["Start Time (Synced)"])
    calib_topics = ["notify.calibration.calibration_data", "notify.accuracy_test.data"]

    lines = iter(pupil_log)
    
    session_start = "notify.calibration.started"
    
    calib_times = []
    movement_times = []
    marker_times = []
    topics = set()
    calib_sessions = []
    
    pupils = [], []

    for line in lines:
        topic, ts, data = line
        data['recv_ts'] = ts
        if topic == "pupil.0":
            pupils[0].append(data)
        if topic == "pupil.1":
            pupils[1].append(data)
        
        #if topic not in topics:
        #    print(topic)
        #    topics.add(topic)
        
        if topic == "notify.calibration.marker_sample_completed":
            calib_times.append(data["timestamp"])
            continue
        if topic == "notify.calibration.marker_moved_too_quickly":
            movement_times.append(data["timestamp"])
            continue
        if topic == "notify.calibration.marker_found":
            marker_times.append(data["timestamp"])
            continue

        if topic not in calib_topics: continue
        
        calib_times = np.array(calib_times)
        marker_times = np.array(marker_times)
        calib_starts = marker_times.searchsorted(calib_times, side='right')
        calib_starts = (marker_times[calib_starts - 1])
        ref_ts = [p['timestamp'] for p in data["ref_list"]]
        ref_marker = calib_times.searchsorted(ref_ts)
        ref_marker[ref_marker >= len(calib_times)] = len(calib_times) - 1

        """
        plt.plot(ref_ts, ref_marker, '.-', color='black')
        for t in calib_times:
            plt.axvline(t)
        for t in movement_times:
            plt.axvline(t, color='red')
        for t in marker_times:
            plt.axvline(t, color='green')
        for t in calib_starts:
            plt.axvline(t, color='black')
        plt.show()
        """
        
        for i in range(len(data["ref_list"])):
            mi = ref_marker[i]
            s, e = calib_starts[mi], calib_times[mi]
            if data["ref_list"][i]['timestamp'] < s:
                mi = np.nan
            data["ref_list"][i]["ref_idx"] = mi
        data['abs_time'] = data["timestamp"] + time_delta
        calib_sessions.append(data)
        calib_times = []
        marker_times = []
        movement_times = []
    return calib_sessions, pupils

def get_matched_calib(data):
    #pupil_list = [p for p in data["pupil_list"] if p["confidence"] > quality_threshold]
    pupil_list = data["pupil_list"]
    eye0 = [p for p in pupil_list if p["id"] == 0]
    eye1 = [p for p in pupil_list if p["id"] == 1]
    ref_list = data["ref_list"]
    #ref_list = reffilt(data["ref_list"])
    #ref_list = [p for p in ref_list if np.isfinite(p['ref_idx'])]
    
    matched_eye0 = calibrate.closest_matches_monocular(ref_list, eye0)
    ref_idx0 = [p['ref']['ref_idx'] for p in matched_eye0]
    confidence0 = [p['pupil']['confidence'] for p in matched_eye0]
    matched_eye0 = calibrate.preprocess_2d_data_monocular(matched_eye0)
    matched_eye0 = np.array(matched_eye0)


    matched_eye1 = calibrate.closest_matches_monocular(ref_list, eye1)
    ref_idx1 = [p['ref']['ref_idx'] for p in matched_eye1]
    confidence1 = [p['pupil']['confidence'] for p in matched_eye1]
    matched_eye1 = calibrate.preprocess_2d_data_monocular(matched_eye1)
    matched_eye1 = np.array(matched_eye1)
    
    # This block undistorts the gaze. Probably shouldn't be
    # used at least yet, as the rest of the pupil pipeline works
    # with distorted coordinates until the surface markers are handled.
    # Also uses -1 to 1 coordinate system instead of 0 to 1
    """
    refs = matched_eye1[:,[2,3]]*screen_size
    refs = camera.unprojectPoints(refs, normalize=False)[:,:-1]
    #refs += 1.0; refs /= 2.0
    matched_eye1[:,[2,3]] = np.arctan(refs)
    refs = matched_eye0[:,[2,3]]*screen_size
    refs = camera.unprojectPoints(refs, normalize=False)[:,:-1]
    #refs += 1.0; refs /= 2.0
    matched_eye0[:,[2,3]] = np.arctan(refs)
    """
    
    matched_eye0 = pd.DataFrame.from_records(matched_eye0, columns=["pupil_x", "pupil_y", "target_x", "target_y"])
    matched_eye0['confidence'] = confidence0
    matched_eye0['target_idx'] = ref_idx0
    
    matched_eye1 = pd.DataFrame.from_records(matched_eye1, columns=["pupil_x", "pupil_y", "target_x", "target_y"])
    matched_eye1['confidence'] = confidence1
    matched_eye1['target_idx'] = ref_idx1

    return matched_eye0, matched_eye1

def get_matched_filtered(data):
    eye0, eye1 = get_matched_calib(data)
    eye0.query("target_idx in @use_targets and confidence > @quality_threshold", inplace=True)
    eye1.query("target_idx in @use_targets and confidence > @quality_threshold", inplace=True)
    
    return eye0.values[:,:4], eye1.values[:,:4]

def get_mappings(calib_sessions):
    all_mappings = []
    for session in calib_sessions:
        eyes = get_matched_filtered(session)
        mappings = []
        for eye in eyes:
            mapper, inliers, params = calibrate.calibrate_2d_polynomial(eye, screen_size=screen_size, threshold=threshold)
            mappings.append(mapper)
        all_mappings.append((session['timestamp'], mappings))
    return all_mappings

def recalibrate_session(pupil_log, outpath):
    if os.path.exists(outpath):
        raise RuntimeError(f"Won't overwrite {outpath}")
    pupil_log = map(json.loads, gzip.open(pupil_log))
    for topic, *_ in pupil_log:
        if topic == "notify.recording.started":
            break
    calib_sessions, pupils = load_session(pupil_log)
    
    mappings = get_mappings(calib_sessions)
    interps = []
    for mt, mapping in mappings:
        for m, pupil in zip(mapping, pupils):
            t, x, y, c, rt = zip(*(
                    (p['timestamp'], *p['norm_pos'], p['confidence'], p['recv_ts'])
                    for p in pupil))
            t = np.array(t)
            c = np.array(c)
            rt = np.array(rt)
            # Be a bit pessimistic on the quality as we'll interpolate
            c = -maximum_filter1d(-c, 4)
            g = np.array(m(np.array([x, y])))
            x, y = g
            interp = interp1d(t,
                    np.array([x, y, c, t - mt, rt]).T,
                    axis=0, bounds_error=False)
            interps.append(interp)
    
    # Interpolate the data to (arbitrarily) match the timestamps
    # of the first pupil signal.
    ts = interps[0].x
    data = np.array([interp(ts) for interp in interps])

    # Extract timestamps, locations and confidences of
    # the different signals
    valid = np.all(np.isfinite(data[:,:,1]), axis=0)
    ts = ts[valid]
    data = data[:,valid]
    pos = data[:,:,:2]
    q = data[:,:,2]

    # TODO: There seems to be around 0.1 s lag and a bit
    # weirdly shaped around 0.05 s jtter between
    # pupil and recv times. Probably want to resynchronize
    # for very time-sensitive analyses.
    rts = data[0,:,4]
    
    # dws is the weight based on how far in time the
    # moment is from the given calibration sessions
    dws = 1.0/np.abs(data[:,:,3])
    dws /= np.sum(dws, axis=0)

    # Multiply the time distance weights with the
    # quality weights and normalize to sum to one
    weights = dws*q
    ws = np.sum(weights, axis=0)
    weights = weights/ws
    mean = np.einsum("eta,et->ta", pos, weights)
    
    # Dump as pldata.
    topic = 'gaze.2d.01'
    
    pack = lambda x: msgpack.packb(x, use_bin_type=True)
    with open(outpath, 'wb') as out:
        for t, rt, pos, conf in zip(ts, rts, mean, ws):
            row = dict(
                    topic=topic,
                    norm_pos=pos.tolist(),
                    confidence=conf,
                    timestamp=t,
                    recv_ts=rt)
            out.write(pack([topic, pack(row)]))
    
    """
    mean[ws < 0.5] = np.nan
    for mq, mp in zip(dws*q*dws.shape[0], pos):
        mp[mq < 0.5] = np.nan
        plt.plot(ts, mp[:,1], alpha=0.3)
    plt.plot(ts, mean[:,1], color='black')
    #mean = np.average(pos, weights=q, axis=1)
    #print(mean)
    #disagreement = np.mean(np.std(pos, axis=0), axis=1)
    #plt.plot(ts, disagreement)
    plt.show()
    """
    
    

if __name__ == '__main__':
    import argh
    argh.dispatch_command(recalibrate_session)
