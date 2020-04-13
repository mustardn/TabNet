import fretboardgtr
from fretboardgtr import chordgtr
import cairosvg
import numpy as np
import svgwrite
from fretboardgtr.fretboardgtr import FretBoardGtr
import imageio

def get_fingering(labels, f, round_labels=False):

    if round_labels:
        labels = np.around(labels)
    fingering=[]
    for string in labels[f]:
        try:
            fingering.append(np.where(string == 1)[0][0])
        except:
            fingering.append(0)

    return fingering


def draw_fretboard(annot, i):
    fingering = []
    for string in annot[i]:
        fingering.append(np.where(string == 1)[0][0])

    if fingering == [0, 0, 0, 0, 0, 0]:
        F = open_fret(fingering=fingering)
    else:
        F = fretboardgtr.ChordGtr(fingering=fingering, root='C')
        F.theme(show_note_name=True, color_chord=False)
        F.draw()
    d = cairosvg.svg2png(F.dwg.tostring())

    return F, d

def open_fret(fingering):

    #this function was created to support an input of [0, 0, 0, 0, 0, 0]
    F = fretboardgtr.ChordGtr()
    F.emptybox()
    F.background_fill()
    F.createfretboard()
    F.nut()
    F.show_tuning(fingering)
    fingname=F.tuning
    inter=FretBoardGtr.find_intervals(fingname,F.root)
    for i in range(0,len(F.tuning),1):
        X=F.wf*(1+i)+F._ol
        Y=F.hf*(fingering[i]+1/2)+F._ol
        color=F.dic_color[inter[i]]
        name_text=fingname[i]
        F.dwg.add(F.dwg.circle((X,Y),r=F.R,fill=F.open_circle_color,stroke=color,stroke_width=F.open_circle_stroke_width))
        t=svgwrite.text.Text(name_text, insert=(X,Y),font_size=F.fontsize_text,font_weight="bold",fill=F.open_text_color,
                             style="text-anchor:middle;dominant-baseline:central")
        F.dwg.add(t)

    return F

def label_results(y_pred, y_gt, f, num_frets=10):

    """
    :param y_pred: predicted labels
    :param y_gt: ground truth labels
    :param f: frame number
    :return: F: the fretboard object, d: PNG output readable by imageio to write video files
    """

    # dictionary of defined colours
    colour_dict = {"red": 'rgb(231, 0, 0)',
                   "yellow": 'rgb(249, 229, 0)',
                   "orange": 'rgb(249, 165, 0)',
                   "green": 'rgb(0, 154, 0)',
                   "navy": 'rgb(0, 15, 65)',
                   "blue": 'rgb(0, 73, 151)',
                   "brown": 'rgb(168, 107, 98)',
                   "pink": 'rgb(222, 81, 108)',
                   "purple": 'rgb(120, 37, 134)',
                   "plum": 'rgb(120, 25, 98)'
                   }

    y_pred = np.around(y_pred)

    fingering_pred = get_fingering(y_pred, f)
    fingering_gt = get_fingering(y_gt, f)

    # initialize fretboardgtr object
    F = fretboardgtr.ChordGtr()

    # draw output box and background fill
    F.dwg = svgwrite.Drawing(F.path,
                             size=(850, (F.hf) * (num_frets + 2) + F._ol),
                             profile='tiny')
    F.dwg.add(F.dwg.rect(insert=(F.wf + F._ol, F.hf + F._ol),
                         size=(850, 850),
                         rx=None, ry=None, fill=F.background_color))

    # draw the fretboard
    createfretboard(F)
    F.nut()
    #F.show_tuning(fingering_gt)
    fingname = F.tuning
    #inter = FretBoardGtr.find_intervals(fingname, F.root)
    fretfing = [0 if v == None else v for v in fingering_gt]

    try:
        minfret = min(v for v in fretfing if v > 0)
    except:
        pass

    # Draw Legend
    X_legend = F.wf * (8) + F._ol
    Y_legend = F.hf * (2) + F._ol

    F.dwg.add(F.dwg.circle((X_legend, Y_legend), r=F.R, fill=colour_dict['green'],
                           stroke=colour_dict['green'], stroke_width=F.open_circle_stroke_width))
    F.dwg.add(svgwrite.text.Text('= Ground truth label', insert=(X_legend + F.wf * 0.5, Y_legend),
                                 font_size=F.fontsize_text, font_weight="bold", fill=F.open_text_color,
                                 style="text-anchor:start;dominant-baseline:central"))

    F.dwg.add(F.dwg.circle((X_legend, Y_legend + F.hf), r=F.R, fill=colour_dict['green'],
                           stroke=colour_dict['yellow'], stroke_width=F.open_circle_stroke_width))
    F.dwg.add(svgwrite.text.Text('= Correctly Predicted', insert=(X_legend + F.wf * 0.5, Y_legend + F.hf),
                                 font_size=F.fontsize_text, font_weight="bold", fill=F.open_text_color,
                                 style="text-anchor:start;dominant-baseline:central"))

    F.dwg.add(F.dwg.circle((X_legend, Y_legend + 2 * F.hf), r=F.R, fill=colour_dict['red'],
                           stroke=colour_dict['red'], stroke_width=F.open_circle_stroke_width))
    F.dwg.add(svgwrite.text.Text('= Incorrectly Predicted', insert=(X_legend + F.wf * 0.5, Y_legend + 2 * F.hf),
                                 font_size=F.fontsize_text, font_weight="bold", fill=F.open_text_color,
                                 style="text-anchor:start;dominant-baseline:central"))

    # Draw predictions and labels on fretboard
    for i in range(0, len(F.tuning), 1):

        # X coordinate varies based on the string number the finger is on
        X = F.wf * (1 + i) + F._ol
        # Y coordinate varies based on the fret number
        Y_gt = F.hf * (fingering_gt[i] + 1 / 2) + F._ol
        Y_pred = F.hf * (fingering_pred[i] + 1 / 2) + F._ol

        if fingering_gt[i] == 0:
            # correct prediction, no finger on the fretboard
            if fingering_pred[i] == 0:
                pass
            else:
                # incorrect prediction
                F.dwg.add(F.dwg.circle((X, Y_pred), r=F.R, fill=colour_dict['red'], stroke=colour_dict['red'],
                                       stroke_width=F.open_circle_stroke_width))

        elif fingering_gt[i] > 0:
            # correct prediction
            if fingering_pred[i] == fingering_gt[i]:
                F.dwg.add(F.dwg.circle((X, Y_gt), r=F.R, fill=colour_dict['green'], stroke=colour_dict['yellow'],
                                       stroke_width=F.open_circle_stroke_width))
            # incorrect prediction of 0, only draw the ground truth
            elif fingering_pred[i] == 0:
                F.dwg.add(F.dwg.circle((X, Y_gt), r=F.R, fill=colour_dict['green'], stroke=colour_dict['green'],
                                       stroke_width=F.open_circle_stroke_width))
            else:
                # incorrect prediction, draw prediction and ground truth
                F.dwg.add(F.dwg.circle((X, Y_pred), r=F.R, fill=colour_dict['red'], stroke=colour_dict['red'],
                                       stroke_width=F.open_circle_stroke_width))
                F.dwg.add(F.dwg.circle((X, Y_gt), r=F.R, fill=colour_dict['green'], stroke=colour_dict['green'],
                                       stroke_width=F.open_circle_stroke_width))

    # convert image to readable PNG format
    d = cairosvg.svg2png(F.dwg.tostring())

    return F, d

def createfretboard(F, num_frets=14):
        '''
        Create an empty set of rectangles based on tunings.
        '''
        fretfing = [0 if v == None else v for v in F.fingering]

        # Creation of fret
        if max(fretfing) > 4:
            for i in range(F.gap + 2):
                # F.gap +2 : two is for the beginning and the end of the fretboard
                F.dwg.add(
                    F.dwg.line(
                        start=(F.wf + F._ol, (F.hf) * (i + 1) + F._ol),
                        end=((F.wf) * (len(F.tuning)) + F._ol, (F.hf) * (1 + i) + F._ol),
                        stroke=F.fretcolor,
                        stroke_width=F.fretsize
                    )
                )
        else:
            for i in range(num_frets):
                # F.gap +1 :  for  the end of the fretboard and (i+2) to avoid first fret when nut
                F.dwg.add(
                    F.dwg.line(
                        start=(F.wf + F._ol, (F.hf) * (i + 2) + F._ol),
                        end=((F.wf) * (len(F.tuning)) + F._ol, (F.hf) * (i + 2) + F._ol),
                        stroke=F.fretcolor,
                        stroke_width=F.fretsize
                    )
                )

        # creation of strings
        if F.string_same_size == False:
            string_size_list = [((F.string_size) - i / 4) for i in range(len(F.tuning))]

        elif F.string_same_size == True:
            string_size_list = [(F.string_size) for i in range(len(F.tuning))]

        for i in range(len(F.tuning)):
            F.dwg.add(
                F.dwg.line(
                    start=((F.wf) * (1 + i) + F._ol, F.hf + F._ol - F.fretsize / 2),
                    end=((F.wf) * (1 + i) + F._ol, F.hf + F._ol + (num_frets) * F.hf + F.fretsize / 2 + F.fretsize),
                    stroke=F.strings_color,
                    stroke_width=string_size_list[i]
                )
            )

def get_video(filename, npzfile=None, y_pred=None, y_gt=None):

    if npzfile:
        labels = np.load(npzfile)
        y_pred = labels['y_pred']
        y_gt = labels['y_gt']

    u, indices = np.unique(y_gt, return_index=True, axis=0)

    frames = []
    for i in sorted(indices):
        F, d = label_results(y_pred, y_gt, i)
        frames.append(d)

    arr_frames = [imageio.imread(d) for d in frames]
    imageio.mimsave(filename, arr_frames, fps=5, quality=7)