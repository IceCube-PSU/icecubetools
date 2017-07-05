from icecube import genie_icetray

from icecube import icetray

from icecube.dataclasses import I3MapStringVectorDouble
from icecube.dataclasses import I3MapStringDouble
from icecube.dataclasses import I3MapStringInt
from icecube.dataclasses import I3MapStringBool
import numpy as np

class ExtractGENIESystematics(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter("GENIEResultDict_Name", "Name of GENIEResult object", None)
        self.AddParameter("Output_Name", "Name of Output object", None)
    def Configure(self):
        self.genie_name = self.GetParameter("GENIEResultDict_Name")
        if self.genie_name is None:
            self.genie_name = "I3GENIEResultDict"
        self.output_name = self.GetParameter("Output_Name")
        if self.output_name is None:
            self.output_name = "GENIE_Systematics"
    def DAQ(self, frame):
        if frame.Has(self.genie_name):
            genie_res = frame[self.genie_name]
            MCweight = frame["I3MCWeightDict"]
            if "GENIEWeight" in MCweight:
                wgt_zero = MCweight["GENIEWeight"]
            else:
                return True
            genie_syst = I3MapStringDouble()
            for key in genie_res.keys():
                if 'rw' in key:
                    syst_values = np.array(genie_res[key])
                    for i in range(0,4):
                        genie_syst[key+'_%i'%i] = syst_values[i]
            frame[self.output_name] = genie_syst
            # somehow there is no '_syststeps' in the keys, so this if statement is useless
            if "_syststeps" in genie_res.keys():
                print "step 5.1"
                genie_syst = I3MapStringVectorDouble()
                i_n_sigma = genie_res["_syststeps"]
                n_i_sigma = dict()
                for i in xrange(0,len(i_n_sigma)):
                    n_i_sigma[i_n_sigma[i]] = i
                sorted_sigma = n_i_sigma.keys()
                sorted_sigma.sort()
                genie_syst["n_sigma"] = sorted_sigma
                for key in genie_res.keys():
                    if 'rw_' in key:
                        syst_values = []
                        orig_syst_values = genie_res[key]
                        all_syst_1 = True
                        for ns in sorted_sigma:
                            new_syst_i = n_i_sigma[ns]
                            if orig_syst_values[new_syst_i] != 1:
                                all_syst_1 = False
                                break
                        for ns in sorted_sigma:
                            new_syst_i = n_i_sigma[ns]
                            if not all_syst_1:
                                ratio = orig_syst_values[new_syst_i]/wgt_zero
                            else:
                                ratio = 1.
                            syst_values.append(ratio)
                        genie_syst[key[3:]] = syst_values
                frame[self.output_name] = genie_syst
        self.PushFrame(frame)
        return True

class ExtractGENIEType(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter("GENIEResultDict_Name", "Name of GENIEResult object", None)
        self.AddParameter("Output_Name", "Name of Output object", None)
    def Configure(self):
        self.genie_name = self.GetParameter("GENIEResultDict_Name")
        if self.genie_name is None:
            self.genie_name = "I3GENIEResultDict"
        self.output_name = self.GetParameter("Output_Name")
        if self.output_name is None:
            self.output_name = "GENIE_InteractionType"
    def DAQ(self, frame):
        if frame.Has(self.genie_name):
            genie_res = frame[self.genie_name]
            genie_type = I3MapStringBool()
            for key in ["cc", "nc", "res", "dis", "qel", "coh", "charm", "dfr", "em", "imd", "nuel", "resid", "sea" ]:
                if not key in genie_res.keys():
                    continue
                genie_type[key] = genie_res[key]
            frame[self.output_name] = genie_type
        self.PushFrame(frame)
        return True

class ExtractGENIEIterTarget(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter("GENIEResultDict_Name", "Name of GENIEResult object", None)
        self.AddParameter("Output_Name", "Name of Output object", None)
    def Configure(self):
        self.genie_name = self.GetParameter("GENIEResultDict_Name")
        if self.genie_name is None:
            self.genie_name = "I3GENIEResultDict"
        self.output_name = self.GetParameter("Output_Name")
        if self.output_name is None:
            self.output_name = "GENIE_InteractionTarget"
    def DAQ(self, frame):
        if frame.Has(self.genie_name):
            genie_res = frame[self.genie_name]
            genie_info = I3MapStringInt()
            for key in [ "A", "Z", "hitnuc", "hitqrk", "tgt"]:
                if not key in genie_res.keys():
                    continue
                genie_info[key] = genie_res[key]
            frame[self.output_name] = genie_info
        self.PushFrame(frame)
        return True

class ExtractGENIEIterInfo(icetray.I3Module):
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter("GENIEResultDict_Name", "Name of GENIEResult object", None)
        self.AddParameter("Output_Name", "Name of Output object", None)
    def Configure(self):
        self.genie_name = self.GetParameter("GENIEResultDict_Name")
        if self.genie_name is None:
            self.genie_name = "I3GENIEResultDict"
        self.output_name = self.GetParameter("Output_Name")
        if self.output_name is None:
            self.output_name = "GENIE_InteractionInfo"
    def DAQ(self, frame):
        if frame.Has(self.genie_name):
            genie_res = frame[self.genie_name]
            genie_info = I3MapStringDouble()
            for key in ["diffxsec", "xsec", "Q2", "Q2s", "W", "Ws", "y", "ys", "x", "xs", "t", "ts" ]:
                if not key in genie_res.keys():
                    continue
                genie_info[key] = genie_res[key]
            frame[self.output_name] = genie_info
        self.PushFrame(frame)
        return True
