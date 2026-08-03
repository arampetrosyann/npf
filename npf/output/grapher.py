import math
import re
import natsort
import copy
import traceback
import sys
import itertools
import os

from sklearn import tree

from npf.output.statistics import Statistics
from npf.output.transform.pandas import to_pandas
from npf.output.transform.combine_variables import combine_variables
from npf.output.transform.result_as_variable import result_as_variable
from npf.output.notebook.notebook import prepare_notebook_export
from npf.output.web.web import prepare_web_export
from npf.models.units import *
from npf.models.units import is_numeric
from npf.models.units import get_bool
from npf.models.units import is_bool
from npf.models.units import get_numeric
import npf.models.units
if sys.version_info < (3, 7):
    from orderedset import OrderedSet
else:
    from ordered_set import OrderedSet

from asteval import Interpreter
from collections import OrderedDict
from typing import List
import numpy as np
from math import log, pow
from matplotlib.ticker import Formatter, NullLocator

from npf.models.series import Series
from npf.models.dataset import Run, group_val, var_divider
from npf.tests.variable import replace_variables
from npf.output.graph.variable_to_series import extract_variable_to_series
from npf.output.graph.series_to_graph import series_to_graph
from npf.output.graph.graphdata import GraphData
from npf.output.graph.auto_vis import plot_graphs_with_lux
import npf

import pandas as pd


def roundf(x, prec):
    exp = pow(10, prec)
    x = round(float(x) * exp)
    x = x / exp
    return x
    
class Map(OrderedDict):
    def __init__(self, fname):
        super().__init__()
        if fname:
          for fn in fname.split('+'):
            f = open(fn, 'r')
            for line in f:
                line = line.strip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue
                k,v = [x.strip() for x in line.split(':',1)]
                self[re.compile(k)] = v

    def search(self, map_v):
        return next((v for k, v in self.items() if re.search(k,str(map_v))), None)


def guess_type(d):
    for k,v in d.items():
        if is_numeric(v):
            d[k]=get_numeric(v)
    return d


def broken_axes_ratio(values):
    if len(values) == 3:
        _, _, ratio = values
        if ratio is not None:
            return ratio
    elif len(values) == 2:
        vmin, vmax = values
        if vmin is not None and vmax is not None:
            return vmax-vmin
    return 1


class Grapher:
    def __init__(self):
        self.scripts = set()
        self._config_cache = {}

    def config_bool(self, var, default=None):
        val = self.config(var, default)
        return get_bool(val)

    def config_bool_or_in(self, var, obj, default=None):
        val = self.config(var, default)

        #If not found, return the default
        if val is None:
            return default

        if type(val) == type(obj) and val == obj:
            return True

        if isinstance(val, list):
            return obj in val
        if is_bool(val):
            return get_bool(val)
        return default

    def config(self, var, default=None):
        for script in self.scripts:
            if var in script.config:
                return script.config[var]
        return default

    def configlist(self, var, default=None):
        for script in self.scripts:
            if var in script.config:
                return script.config.get_list(var)
        return default

    def configdict(self, var, default=None):
        for script in self.scripts:
            if var in script.config:
                return script.config.get_dict(var)
        return default

    def scriptconfig(self, var, key, default=None, result_type=None):
        if (var,key,result_type) in self._config_cache:
            return self._config_cache[(var,key,result_type)]
        else:
            v = self._scriptconfig(var, key, default, result_type)
            self._config_cache[(var,key,result_type)] = v
            return v

    def _scriptconfig(self, var, key, default, result_type):
        for script in self.scripts:
            if var in script.config:
                return script.config.get_dict_value(var,key,default=default,result_type=result_type)
        return default

    def result_in_list(self, var, result_type):
        l = self.configlist(var, [])
        return (result_type in l) or ("result-" + result_type in l) or ("result" in l)

    def var_name(self, key, result_type=None) -> str:
        return self.scriptconfig("var_names", key, key if not result_type else result_type, result_type)

    def us(self, x, pos):
        return self.formats(x, pos, 1)

    def formats(self,x,pos,mult):
        return "%d" % (x * mult)

    def get_var_lim(self, key, result_type):
        var_lim = self.scriptconfig("var_lim", key, result_type=result_type, default=None)
        axes = []
        if var_lim:
          for var_lim in var_lim.split('+'):
            ymin = None
            ymax = None
            ratio = None

            if var_lim.startswith('-'):
                n = var_lim[1:].split('-',2)
                n[0] = "-"+n[0]
            else:
                n = var_lim.split('-',2)
            try:
              # Try to read the ratio, if given
              if len(n) == 3 and n[1] != "" and  n[2] != "":
                ymin, ymax, ratio = (parseUnit(x) for x in n)
              if len(n) >= 2 and n[1] != "":
                ymin, ymax = (parseUnit(x) for x in n[:2])
              else:
                f=float(n[0])
                if f==0:
                    ymin=f
                else:
                    ylim=f
            except Exception as e:
                print(e)
                traceback.print_exc()
            if ratio:
                axes.append([ymin,ymax,ratio])
            else:
                axes.append([ymin,ymax])
        else:
            ymin = None
            ymax = None

            axes.append([ymin,ymax])
        return axes


    class ByteFormatter(Formatter):
        def __init__(self,unit,ps="",compact=False,k=1000,mult=1):
            self.unit = unit
            self.ps = ps
            if unit == "Bps":
                self.unit = "B"
                self.ps = "/s"

            self.compact = compact
            self.k = k
            self.mult = mult

        def __call__(self, x, pos=None):
            """
            Return the value of the user defined function.

            `x` and `pos` are passed through as-is.
            """
            return self.formatb(x * self.mult, pos,self.unit,self.ps,self.compact,self.k)

        def formatb(self, x, pos, unit, ps, compact, k):
            if compact:
                pres="%d"
            else:
                pres="%.2f"
            if x >= (k*k*k):
                v = x / (k*k*k)
                if v < 10 and compact:
                    pres = "%.1f"
                return (pres + "G%s"+ps) % (v, unit)
            elif x >= (k*k):
                return (pres + "M%s"+ps) % (x / (k*k), unit)
            elif x >= k:
                return (pres+ "K%s"+ps) % (x / k, unit)
            else:
                return (pres+ "%s"+ps) % (x, unit)

    def aggregate_variable(self, key:str, series:List[Series], method:str):
        """Aggregate all values of a variables with some method

        Args:
            key: Key (variable name) to combine
            series: The list of series
            method: The method to use for merging variable (see group_val)

        Returns:
            The new list of series
        """
        nseries = []
        for i,(test, build, all_results) in enumerate(series):
            aggregates = OrderedDict()
            for run, run_results in all_results.items():
                    #                    if (graph_variables and not run in graph_variables):
                    #                        continue
                newrun = run.copy()
                #Remove all aggregated variables from run's variables
                for k in key.split("+"):
                    if k in newrun.write_variables():
                        del newrun.write_variables()[k]

                newrun.write_variables()[key] = 'AGG'

                #Add all run_results as a list for the common variables (merging the aggregated ones)
                aggregates.setdefault(newrun,[]).append(run_results)

            new_all_results = OrderedDict()

            #Now we have to actually aggregate the results from all merge list
            try:
              for run, multiple_run_results in aggregates.items():
                    #multiple_run_results is a list of results for each run
                    new_run_results = OrderedDict()
                    #agg becomes a dict of result_type (for this run) to all the results for this run
                    agg = {}
                    all_result_types = set()
                    for run_results in multiple_run_results:
                        for result_type, results in run_results.items():
                            agg.setdefault(result_type,{})
                            all_result_types.add(result_type)
                            for i, result in enumerate(results):
                                agg[result_type].setdefault(i,[]).append(result)
                    for result_type in all_result_types:
                        if method == 'all':
                            new_run_results[result_type] = list(itertools.chain.from_iterable([ag for i,ag in agg[result_type].items()]))
                        else:
                            new_run_results[result_type] = list(filter(lambda x:  not np.isnan(x), [group_val(np.asarray(ag), method, lambda fl: agg[fl][i]) for i,ag in agg[result_type].items()]))
                    new_all_results[run] = new_run_results
            except Exception as e:
                    print("ERROR : Aggregate caneclled because an error occured, ", e)
            nseries.append((test,build,new_all_results))
        return nseries

    # Extract the variable key so it becomes a serie
    def extract_variable_to_series(self, key, vars_values, all_results, dyns, build, script) -> GraphData:
        if not key in dyns:
            raise ValueError("Cannot extract %s because it is not a dynamic variable (%s are)" % (key, ', '.join(dyns)))
        dyns.remove(key)
        series = []
        versions = []
        values = list(vars_values[key])
        del vars_values[key]
        try:
            #values.sort()
            pass
        except TypeError:
            print("ERROR : Cannot sort the following values :", values)
            return

        for i, value in enumerate(values):
            newserie = OrderedDict()
            for run, run_results in all_results.items():
                #                    if (graph_variables and not run in graph_variables):
                #                        continue
                if run.read_variables()[key] == value:
                    newrun = run.copy()
                    del newrun.write_variables()[key]
                    newserie[newrun] = run_results

            if type(value) is tuple:
                value = value[1]
            versions.append(value)
            nb = build.copy()
            nb._pretty_name = str(value)
            if len(self.graphmarkers) > 0:
                nb._marker = self.graphmarkers[i % len(self.graphmarkers)]
            series.append((script, nb, newserie))
            self.glob_legend_title = self.var_name(key)

        if len(dyns) == 1:
            key = dyns[0]
            do_sort = True
        elif len(dyns) == 0:
            do_sort = True
        else:
            key = "Variables"
            do_sort = False
        do_sort = self.config_bool_or_in('graph_x_sort', key, default=do_sort)

        graph = GraphData(self)
        graph.do_sort = do_sort
        graph.key = key

        graph.set_series(series, vars_values)
        return graph

    # Convert a list of series to a graph object
    #  if the list has a unique item and there are dynamic variables, one
    #  dynamic variable will be extracted to make a list of serie
    def series_to_graph(self, series, dyns, vars_values):
        nseries = len(series)

        ndyn = len(dyns)
        if self.options.do_transform and (nseries == 1 and ndyn > 0 and not self.options.graph_no_series and not (
                            ndyn == 1 and all_num(vars_values[dyns[0]]) and len(vars_values[dyns[0]]) > 2) and dyns[0] != "time"):
            """Only one serie: expand one dynamic variable as serie, but not if it was plotable as a line"""
            script, build, all_results = series[0]
            if self.config("var_serie") and self.config("var_serie") in dyns:
                key = self.config("var_serie")
            else:
                key = None
                # First pass : use the non-numerical variable with the most points, but limited to 10
                n_val = 0
                nonums = []
                for i in range(ndyn):
                    k = dyns[i]
                    if k == 'time':
                        continue
                    if not all_num(vars_values[k]):
                        nonums.append(k)
                        if len(vars_values[k]) > n_val and len(vars_values[k]) < 10:
                            key = k
                            n_val = len(vars_values[k])
                if key is None:
                    # Second pass if that missed, use the numerical variable with the less point if dyn==2 (->lineplot) else the most points
                    n_val = 0 if ndyn > 2 else 999
                    for i in range(ndyn):
                        k = dyns[i]
                        if k == 'time':
                            continue
                        if (ndyn > 2 and len(vars_values[k]) > n_val) or (ndyn <= 2 and len(vars_values[k]) < n_val):
                            key = k
                            n_val = len(vars_values[k])

            # Key is found, no the extraction itself
            if not key:
                key = 'time'
            if key:
                graph = self.extract_variable_to_series(
                                key=key,
                                vars_values=vars_values,
                                all_results=all_results,
                                dyns=dyns,
                                build=build,
                                script=script)

        else:
            self.glob_legend_title = None
            if ndyn == 0:
                key = "version"
                do_sort = False
            elif ndyn == 1:
                key = dyns[0]
                do_sort = True
            else:
                key = "Variables"
                do_sort = False
            graph = GraphData(self)
            graph.key = key
            graph.do_sort = do_sort
            graph.set_series(series, vars_values)
        return graph

    def map_variables(self, map_k, fmap, series, vars_values):
            transformed_series = []
            for i, (test, build, all_results) in enumerate(series):
                new_results={}
                for run, run_results in all_results.items():
                    if map_k and not map_k in run.variables:
                        new_results[run] = run_results
                        continue
                    map_v = run.variables[map_k]
                    new_v = fmap.search(map_v)
                    if new_v:
                        if map_v in vars_values[map_k]:
                            vars_values[map_k].remove(map_v)
                        if new_v and new_v != " ":
                            run.variables[map_k] = new_v
                            vars_values[map_k].add(new_v)
                            if run in new_results:
                              for result_type, results in new_results[run].items():
                                nr = run_results[result_type]
                                for i in range(min(len(results),len(nr))):
                                    results[i] += nr[i]
                            else:
                              new_results[run] = run_results
                    else:
                        new_results[run] = run_results
                transformed_series.append((test, build, new_results))
            return transformed_series

    def graph(self, filename, options, fileprefix=None, graph_variables: List[Run] = None, title=False, series:Series=None, return_fig=False):
        """
        The function "graph" is used to create a graph based on the given parameters and save it to a
        file.

        :param filename: The filename parameter is a string that represents the name of the file where
        the graph will be saved
        :param options: The "options" parameter is a dictionary that contains various options for
        customizing the graph. It is the result of the parsing of the arguments, equal to npf.options.
        It is passed for legacy reasons.
        :param fileprefix: The `fileprefix` parameter is an optional prefix that can be added to the
        filename of the graph. It is useful when you want to differentiate between multiple graphs that
        are being generated. If `fileprefix` is provided, it will be added to the beginning of the
        filename followed by an underscore
        :param graph_variables: The `graph_variables` parameter is a list of `Run` objects. Each `Run`
        object represents a set of data points that will be plotted on the graph. If none, all observations
        will be shown.
        :type graph_variables: List[Run]
        :param title: The `title` parameter is a boolean value that determines whether or not to display
        a title for the graph. If `title` is set to `True`, a title will be displayed on the graph. If
        `title` is set to `False`, no title will be displayed, defaults to False (optional)
        :param series: The "series" parameter is used to specify the data series that will be plotted on
        the graph. It is typically a list of values or a list of lists, where each value or sublist
        represents a data series. Each data series will be plotted as a separate line or bar on the
        graph
        """
        self.options = options
        self.return_fig = return_fig
        if self.options.graph_size is None:
            self.options.graph_size = [6.4, 4.8]
        if series is None:
            series = []


        for test, _, _ in series:
                self.scripts.add(test)

        # If no graph variables, use the first serie
        if graph_variables is None:
            graph_variables = OrderedSet()
            for serie in series:
                for run, results in serie[2].items():
                    graph_variables.add(run)

        if not series:
            print("No data...")
            return

        # Add series to a pandas dataframe
        if options.pandas_filename is not None or options.web is not None or options.notebook_path is not None:
            all_results_df = to_pandas(series)

            # Save the pandas dataframe into a csv
            if options.pandas_filename is not None:
                pandas_df_name=os.path.splitext(options.pandas_filename)[0] + ("-"+fileprefix if fileprefix else "") + ".csv"
                # Create the destination folder if it doesn't exist
                df_path = os.path.dirname(pandas_df_name)
                if df_path and not os.path.exists(df_path):
                    os.makedirs(df_path)

                all_results_df.to_csv(pandas_df_name, index=True, index_label="index", sep=",", header=True)
                print("Pandas dataframe written to %s" % pandas_df_name)

        #Overwrite markers and lines from user
        self.graphmarkers = self.configlist("graph_markers")
        self.graphlines = self.configlist("graph_lines")

        # Combine variables as per the graph_combine_variables config parameter
        # for instance if A has [1,2] values and B has [yes,no], graph_combine_variables={A+B:Unique}
        # will remove A and B, and create a new Unique variables with ["1, yes", "1, no", "2, yes", "2, no"]
        for tocombine in self.configlist('graph_combine_variables', []):
            series, graph_variables = combine_variables(series, tocombine, graph_variables)

        # Data transformation : reject outliers if option is given, transform list to arrays, filter according to graph_variables
        #   and divide results as per the var_divider
        filtered_series = []
        vars_values = OrderedDict()
        for i, (test, build, all_results) in enumerate(series):
            new_results = OrderedDict()
            for run, run_results in all_results.items():
                if run in graph_variables:
                    for result_type, results in run_results.items():
                        if self.options.graph_reject_outliers:
                            results = self.reject_outliers(np.asarray(results), test)
                        else:
                            results = np.asarray(results)

                        if self.options.graph_select_max:
                            results = np.sort(results)[-self.options.graph_select_max:]

                        ydiv = var_divider(test, "result", result_type)
                        if np.all(np.isnan(results)):
                            results=np.asarray([0])
                        new_results.setdefault(run.copy(), OrderedDict())[result_type] = results / ydiv

                    for k, v in run.read_variables().items():
                        vars_values.setdefault(k, OrderedSet()).add(v)

            if new_results:
                if len(self.graphmarkers) > 0:
                    build._marker = self.graphmarkers[i % len(self.graphmarkers)]
                filtered_series.append((test, build, new_results))
            else:
                print("No valid data for %s" % build)
        series = filtered_series

        if len(series) == 0:
            return

        #If graph_series_as_variables, take the series and make them as variables
        if self.config_bool('graph_series_as_variables',False):
            new_results = {}
            vars_values['serie'] = set()
            for test, build, all_results in series:
                for run, run_results in all_results.items():
                    run.variables['serie'] = build.pretty_name()
                    vars_values['serie'].add(build.pretty_name())
                    new_results[run] = run_results
            series = [(test, build, new_results)]

        # Transform results to variables as the graph_result_as_variable config
        #  option. It is a dict in the format
        #  a+b+c:var_name[-result_name]
        #  i.e. the first is a + separated list of result and the second member
        #  a new name for the combined variable
        # or
        # a-(.*):var_name[-result_name]
        # Both will create a variable with a/b/c as values or all regex mateched values
        # Example:
        # Values for one run:
        # RESULT-CPU-0 53
        # RESULT-CPU-1 72
        # With CPU-(.*):LOAD it will create two runs
        # CPU=0 -> LOAD = 53
        # CPU=1 -> LOAD = 72
        for result_types, var_name in self.configdict('graph_result_as_variable', {}).items():
            var_unit = self.scriptconfig("var_unit", var_name, default="")
            exploded_series, exploded_vars_values = result_as_variable(series, result_types, var_name, vars_values, var_unit = var_unit)
            self.graph_group(series=exploded_series, vars_values=exploded_vars_values, filename=filename, fileprefix = fileprefix, title=title)


        ret = self.graph_group(series, vars_values, filename=filename, fileprefix = fileprefix, title=title)

        # Export to web format
        if options.web is not None:
            prepare_web_export(series, all_results_df, options.web)

        # Export to Jupyter notebook
        if options.notebook_path is not None:
            prepare_notebook_export(series, all_results_df, self.options, self.config)

        return ret


    def graph_group(self, series, vars_values, filename, fileprefix, title):
        if len(series) == 0:
            print("No valid series...")
            return

        # List of static variables to use in filename
        statics = {}

        # Set lines types
        for i, (script, build, all_results) in enumerate(series):
            build._line = self.graphlines[i % len(self.graphlines)]
            build.statics = {}

        # graph_variables_as_series will force a variable to be considered as
        # a serie. This is different from var_serie which will define
        # what variable to use as a serie when there is only one serie
        for to_get_out in self.configlist('graph_variables_as_series', []):
            try:
                values = natsort.natsorted(vars_values[to_get_out], lambda x: x[1] if type(x) is tuple else x)
            except KeyError as e:
                print("WARNING : Unknown variable %s to export as serie" % to_get_out)
                print("Known variables : ",", ".join(vars_values.keys()))
                continue
            if len(values) == 1:
                statics[to_get_out] = list(values)[0]
            del vars_values[to_get_out]

            transformed_series = []
            for sindex, (test, build, all_results) in enumerate(series):
                new_series = OrderedDict()
                for value in values:
                    new_series[value] = OrderedDict()

                for run, run_results in all_results.items():
                    variables = run.variables.copy()
                    new_run_results = {}
                    value = variables[to_get_out]
                    del variables[to_get_out]
                    new_series[value][Run(variables)] = run_results

                for i, (value, data) in enumerate(new_series.items()):
                    nbuild = build.copy()
                    nbuild.statics = build.statics.copy()
                    nbuild._pretty_name = ' - '.join(([nbuild.pretty_name()] if len(series) > 1 or self.options.show_serie else []) + [ (str(value[1]) if not self.config_bool("graph_variables_explicit") else ("%s = %s" % (self.var_name(to_get_out), str(value[1]))) ) if type(value) is tuple else ("%s = %s" % (self.var_name(to_get_out), str(value)))  ])
                    if len(self.graphmarkers) > 0:
                        nbuild._marker = self.graphmarkers[i % len(self.graphmarkers)]
                    if len(series) == 1: #If there is one serie, expand the line types
                        nbuild._line = self.graphlines[sindex % len(self.graphlines)]

                    nbuild._color_index = sindex + 1
                    nbuild.statics[to_get_out] = value
                    transformed_series.append((test, nbuild, data))

            series = transformed_series

        #Map and combine variables values
        for map_k, fmap in self.configdict('graph_map',{}).items():
            fmap = Map(fmap)
            series = self.map_variables(fmap=fmap, map_k=map_k, series=series, vars_values=vars_values)

        m = self.configdict('graph_map_inline',{})
        if m:
            fmap = Map(None)
            fmap.update(m)
            for k in vars_values.keys():
                series = self.map_variables(fmap=fmap, map_k=str(k), series=series, vars_values=vars_values)

        #round values of a variable to a given precision, if it creates a merge, the list is appended
        for var, prec in self.configdict("var_round",{}).items():
            transformed_series = []
            prec = float(prec)
            for i, (test, build, all_results) in enumerate(series):
                new_all_results = OrderedDict()
                for run, run_results in all_results.items():
                    if var in run.variables:
                        v = roundf(run.variables[var], prec)
                        run.variables[var] = v
                        if run in new_all_results:
                            np.append(new_all_results[run], run_results)
                        else:
                            new_all_results[run] = run_results
                transformed_series.append((test, build, new_all_results))
            series = transformed_series


        # Apply a mathematical formula to results
        for result_type, fil in self.configdict("graph_filter",{}).items():
            transformed_series = []
            aeval = Interpreter()
            for i, (test, build, all_results) in enumerate(series):
                new_all_results = OrderedDict()
                def lam(x):
                    aeval.symtable['x'] = x
                    return aeval(fil)
                for run, run_results in all_results.items():
                    if result_type in run_results:
                        run_results[result_type] = list(filter(lam, run_results[result_type]))
                    new_all_results[run] = run_results
                transformed_series.append((test, build, new_all_results))
            series = transformed_series

        for key, method in self.configdict('var_aggregate').items():
            series = self.aggregate_variable(key=key,series=series,method=method)
            for k in key.split('+'):
                vars_values[k] = ['AGG']

        dyns = []
        for k, v in vars_values.items():
            if len(v) <= 0:
                print("ERROR: Variable %s has no values" % k)
            elif len(v) == 1:
                statics[k] = list(v)[0]
            else:
                dyns.append(k)

        if len(dyns) > int(self.config("graph_max_variables", 2)):

            print("WARNING: Too many variables to plot !")
            keep_vars = []
            if hasattr(self.options, 'graph_keep_variables') and self.options.graph_keep_variables:
                keep_vars = self.options.graph_keep_variables
            else:
                keep_vars = self.configlist("graph_keep_variables", [])

            maxvar = 0
            most_useless = [v for v in reversed(dyns) if v not in keep_vars]
            try:

                for i, (test, build, all_results) in enumerate(series):
                        dataset = Statistics.buildDataset(all_results, test)
                        clf = tree.DecisionTreeRegressor()
                        _, X, y, dtype = dataset[0]
                        clf = clf.fit(X, y)
                        disp = np.var(y)/np.mean(y)
                        if disp > maxvar:
                            maxvar = disp
                            most_useless = [dtype["names"][i] for i in np.argsort(clf.feature_importances_) if dtype["names"][i] in dyns and dtype["names"][i] not in keep_vars]
            except Exception as e:
                    print("ERROR: Could not compute feature importance to ignore most meaningless variable...")
                    print(e)
            while len(dyns) > 2 and len(most_useless) > 0:
                print(f"Variable {most_useless[0]} will be ignored. All points for its various levels will be displayed as variance of the other points.")
                vars_values[most_useless[0]] = ['AGG']
                dyns.remove(most_useless[0])
                series = self.aggregate_variable(key=most_useless[0], series=series, method="all")
                del most_useless[0]
            print(dyns)

        #Divide a serie by another
        prop = self.config('graph_series_prop')
        if prop:
            if len(series) > 1:
                series = GraphData.series_prop(series, prop, self.configdict('graph_cross_reference').values())
                prop = False

        #Eventually overwrite label of series
        graph_series_label = self.config("graph_series_label")
        sv = self.config('graph_subplot_variable', None)
        graphs = []

        #Interpret title
        if title:
            v = {}
            v.update(statics)
            title=replace_variables(v, title)

        # Lux should see Run variables as-is. series_to_graph /
        # extract_variable_to_series promote a dyn var into series identity and
        # delete it from each Run — keep a pre-extraction copy for Lux only.
        f_series = series

        # If a subplot variable is defined, extract it as a serie
        if sv: #Only one supported for now
            graphs = [ None for _ in vars_values[sv] ]
            for j,(script, build, all_results) in enumerate(series):
                graph = extract_variable_to_series(self, sv, vars_values.copy(), all_results, dyns.copy(), build, script)

                self.glob_legend_title = title #This variable has been extracted, the legend should not be the variable name in this case

                if graph_series_label:
                    for i, (test, build, all_results) in enumerate(series):
                        v = {}
                        v.update(statics)
                        v.update(build.statics)
                        build._pretty_name=replace_variables(v, graph_series_label)

#                graph.title = title if title else self.var_name(sv)
#                if len(series) > 1:
#                    graph.title = build._pretty_name + " - " + graph.title
                s = graph.series.copy()
                for i,stuple in enumerate(s):
                    if graphs[i] is None:
                        graphs[i] = copy.copy(graph)
                        graphs[i].title = self.var_name(sv) + " = " + stuple[1]._pretty_name
                        graphs[i].series = list()

                    graphs[i].series.append(stuple)
                    graphs[i].series[j][1]._pretty_name = build._pretty_name
                assert(not sv in graph.vars_values)


            del dyns
            del vars_values
        else:
            graph = series_to_graph(self, series, dyns, vars_values)
            graph.title = title
            graphs.append(graph)

        if prop:
            for graph in graphs:
                graph.series_prop(prop, self.configdict('graph_cross_reference').values())

        if len(graphs) > 0:
            return self.plot_graphs(graphs, filename, fileprefix, f_series=f_series)

    def plot_graphs(self, graphs, filename, fileprefix, f_series=None):
        """
        Render graphs via Lux (Vega-Lite backend).

        Data transforms remain in graph()/graph_group(); this method only
        turns GraphData into Lux Vis → Vega-Lite → PDF/PNG.
        When f_series is provided, Lux uses that (pre series_to_graph) data
        so variables promoted to series identity remain real columns.
        """
        return plot_graphs_with_lux(self, graphs, filename, fileprefix, f_series=f_series)

    def reject_outliers(self, result, test):
        return test.reject_outliers(result)

    def get_show_values(self):
        prec = self.config('graph_show_values', False)

        if not prec:
            return False
        if is_numeric(prec):
            return get_numeric(prec)
        if type(prec) is list and prec and is_numeric(prec[0]):
            return get_numeric(prec[0])
        return 2

