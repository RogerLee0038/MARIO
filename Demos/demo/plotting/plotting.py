# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import string
import hashlib
import warnings
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nevergrad.common.typing as tp
import utils
from exporttable import export_table

# pylint: disable=too-many-locals


_DPI = 250
pure_algorithms = []

# %% Basic tools


def compactize(name: str) -> str:
    if len(name) < 70:
        return name
    hashcode = hashlib.md5(bytes(name, "utf8")).hexdigest()
    name = re.sub(r"\([^()]*\)", "", name)
    mid = 35
    name = name[:mid] + hashcode + name[-mid:]
    return name


def _make_style_generator() -> tp.Iterator[str]:
    lines = itertools.cycle(["-", "--", ":", "-."])  # 4
    markers = itertools.cycle("ov^<>8sp*hHDd")  # 13
    colors = itertools.cycle("bgrcmyk")  # 7
    return (l + m + c for l, m, c in zip(lines, markers, colors))


class NameStyle(tp.Dict[str, tp.Any]):
    """Provides a style for each name, and keeps to it"""

    def __init__(self) -> None:
        super().__init__()
        self._gen = _make_style_generator()

    def __getitem__(self, name: str) -> tp.Any:
        if name not in self:
            super().__setitem__(name, next(self._gen))
        return super().__getitem__(name)


def _make_winners_df(df: pd.DataFrame, all_optimizers: tp.List[str]) -> utils.Selector:
    """Finds mean loss over all runs for each of the optimizers, and creates a matrix
    winner_ij = 1 if opt_i is better (lower loss) then opt_j (and .5 for ties)
    """
    if not isinstance(df, utils.Selector):
        df = utils.Selector(df)
    all_optim_set = set(all_optimizers)
    assert all(x in all_optim_set for x in df.unique("optimizer_name"))
    assert all(x in df.columns for x in ["optimizer_name", "loss"])
    winners = utils.Selector(index=all_optimizers, columns=all_optimizers, data=0.0)
    grouped = df.loc[:, ["optimizer_name", "loss"]].groupby(["optimizer_name"]).mean()
    df_optimizers = list(grouped.index)
    values = np.array(grouped)
    diffs = values - values.T
    # loss_ij = 1 means opt_i beats opt_j once (beating means getting a lower loss/regret)
    winners.loc[df_optimizers, df_optimizers] = (diffs < 0) + 0.5 * (diffs == 0)
    return winners


def aggregate_winners(
    df: utils.Selector, categories: tp.List[str], all_optimizers: tp.List[str]
) -> tp.Tuple[utils.Selector, int]:
    """Computes the sum of winning rates on all cases corresponding to the categories

    Returns
    -------
    Selector
        the aggregate
    int
        the total number of cases
    """
    if not categories:
        return _make_winners_df(df, all_optimizers), 1
    subcases = df.unique(categories[0])
    if len(subcases) == 1:
        return aggregate_winners(df, categories[1:], all_optimizers)
    iterdf, iternum = zip(
        *(
            aggregate_winners(
                df.loc[
                    df.loc[:, categories[0]] == val
                    if categories[0] != "budget"
                    else df.loc[:, categories[0]] <= val
                ],
                categories[1:],
                all_optimizers,
            )
            for val in subcases
        )
    )
    return sum(iterdf), sum(iternum)  # type: ignore


def _make_sorted_winrates_df(victories: pd.DataFrame) -> pd.DataFrame:
    """Converts a dataframe counting number of victories into a sorted
    winrate dataframe. The algorithm which performs better than all other
    algorithms comes first. When you do not play in a category, you are
    considered as having lost all comparisons in that category.
    """
    assert all(x == y for x, y in zip(victories.index, victories.columns))
    winrates = victories / (victories + victories.T).max(axis=1)
    # mean_win = winrates.quantile(.05, axis=1).sort_values(ascending=False)
    mean_win = winrates.mean(axis=1).sort_values(ascending=False)
    return winrates.loc[mean_win.index, mean_win.index]


# %% plotting functions


def remove_errors(df: pd.DataFrame) -> utils.Selector:
    df = utils.Selector(df)
    if "error" not in df.columns:  # backward compatibility
        return df  # type: ignore
    # errors with no recommendation
    nandf = df.select(loss=np.isnan)
    for row in nandf.itertuples():
        msg = f'Removing "{row.optimizer_name}"'
        msg += f" with dimension {row.dimension}" if hasattr(row, "dimension") else ""
        msg += f': got error "{row.error}"' if isinstance(row.error, str) else "recommended a nan"
        warnings.warn(msg)
    # error with recorded recommendation
    handlederrordf = df.select(error=lambda x: isinstance(x, str) and x, loss=lambda x: not np.isnan(x))
    for row in handlederrordf.itertuples():
        warnings.warn(
            f'Keeping non-optimal recommendation of "{row.optimizer_name}" '
            f'with dimension {row.dimension if hasattr(row, "dimension") else "UNKNOWN"} which raised "{row.error}".'
        )
    err_inds = set(nandf.index)
    output = df.loc[[i for i in df.index if i not in err_inds], [c for c in df.columns if c != "error"]]
    # cast nans in loss to infinity
    df.loc[np.isnan(df.loss), "loss"] = float("inf")
    #
    assert (
        not output.loc[:, "loss"].isnull().values.any()
    ), "Some nan values remain while there should not be any!"
    output = utils.Selector(output.reset_index(drop=True))
    return output  # type: ignore


class PatternAggregate:
    def __init__(self, pattern: str) -> None:
        self._pattern = pattern

    def __call__(self, df: pd.Series) -> str:
        return self._pattern.format(**df.to_dict())


_PARAM_MERGE_PATTERN = "{optimizer_name},{parametrization}"


def merge_optimizer_name_pattern(
    df: utils.Selector, pattern: str, merge_parametrization: bool = False, remove_suffix: bool = False
) -> utils.Selector:
    """Merge the optimizer name with other descriptors based on a pattern
    Nothing happens if merge_parametrization is false and pattern is empty string
    """
    if merge_parametrization:
        if pattern:
            raise ValueError(
                "Cannot specify both merge-pattern and merge-parametrization "
                "(merge-parametrization is equivalent to merge-pattern='{optimizer_name},{parametrization}')"
            )
        pattern = _PARAM_MERGE_PATTERN
    if not pattern:
        return df
    df = df.copy()
    okey = "optimizer_name"
    elements = [tup[1] for tup in string.Formatter().parse(pattern) if tup[1] is not None]
    assert okey in elements, (
        f"Missing optimizer key {okey!r} in merge pattern.\nEg: "
        + 'pattern="{optimizer_name}_{parametrization}"'
    )
    others = [x for x in elements if x != okey]
    aggregate = PatternAggregate(pattern)
    sub = df.loc[:, elements].fillna("")
    if len(sub.unique(others)) > 1:
        for optim in sub.unique(okey):
            inds = sub.loc[:, okey] == optim
            if len(sub.loc[inds, :].unique(others)) > 1:
                df.loc[inds, okey] = sub.loc[inds, elements].agg(aggregate, axis=1)
    if remove_suffix:
        df["optimizer_name"] = df["optimizer_name"].replace(r"[0-9\.\-]*$", "", regex=True)
    return df.drop(columns=others)  # type: ignore


def normalized_losses(df: pd.DataFrame, descriptors: tp.List[str]) -> utils.Selector:
    df = utils.Selector(df.copy())
    cases = df.unique(descriptors)
    if not cases:
        cases = [()]
    # Average normalized plot with everything.
    for case in cases:
        subdf = df.select_and_drop(**dict(zip(descriptors, case)))
        losses = np.array(subdf.loc[:, "loss"])
        m = min(losses)
        M = max(losses[losses < float("inf")])
        df.loc[subdf.index, "loss"] = (df.loc[subdf.index, "loss"] - m) / (M - m) if M != m else 1
    return df  # type: ignore


# pylint: disable=too-many-statements,too-many-branches
def create_plots(
    df: pd.DataFrame,
    output_folder: tp.PathLike,
    max_combsize: int = 1,
    xpaxis: str = "budget",
    competencemaps: bool = False,
) -> None:
    """Saves all representing plots to the provided folder

    Parameters
    ----------
    df: pd.DataFrame
        the experiment data
    output_folder: PathLike
        path of the folder where the plots should be saved
    max_combsize: int
        maximum number of parameters to fix (combinations) when creating experiment plots
    xpaxis: str
        x-axis for xp plots (either budget or pseudotime)
    """
    assert xpaxis in ["budget", "pseudotime"]
    df = remove_errors(df)
    df.loc[:, "loss"] = pd.to_numeric(df.loc[:, "loss"])
    df = df.loc[:, [x for x in df.columns if not x.startswith("info/")]]
    # Normalization of types.
    for col in df.columns:
        if col in (
            "budget",
            "num_workers",
            "dimension",
            "useful_dimensions",
            "num_blocks",
            "block_dimension",
            "num_objectives",
        ):
            try:
                df[col] = df[col].astype(float).astype(int)
            except Exception as e1:
                try:
                    for i in range(len(df[col])):
                        float(df[col][i])
                except Exception as e2:
                    assert False, f"Fails at row {i+2}, Exceptions: {e1}, {e2}"
        elif col != "loss":
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(r"\.[0]*$", "", regex=True)
            try:
                df.loc[:, col] = pd.to_numeric(df.loc[:, col])
            except:
                pass
    if "num_objectives" in df.columns:
        df = df[df.num_objectives != 0]  # the optimization did not even start
    # If we have a descriptor "instrum_str",
    # we assume that it describes the instrumentation as a string,
    # that we should include the various instrumentations as distinct curves in the same plot.
    # So we concat it at the end of the optimizer name, and we remove "parametrization"
    # from the descriptor.
    if "instrum_str" in set(df.columns):
        df.loc[:, "optimizer_name"] = df.loc[:, "optimizer_name"] + df.loc[:, "instrum_str"]
        df = df.drop(columns="instrum_str")
        df = df.drop(columns="dimension")
        if "parametrization" in set(df.columns):
            df = df.drop(columns="parametrization")
    df = utils.Selector(df.fillna("N-A"))  # remove NaN in non score values
    assert not any("Unnamed: " in x for x in df.columns), f"Remove the unnamed index column:  {df.columns}"
    assert "error " not in df.columns, f"Remove error rows before plotting"
    required = {"optimizer_name", "budget", "loss", "elapsed_time", "elapsed_budget"}
    missing = required - set(df.columns)
    assert not missing, f"Missing fields: {missing}"
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    # check which descriptors do vary
    descriptors = sorted(
        set(df.columns) - (required | {"instrum_str", "seed", "pseudotime"})
    )  # all other columns are descriptors
    to_drop = [x for x in descriptors if len(df.unique(x)) == 1]
    df = utils.Selector(df.loc[:, [x for x in df.columns if x not in to_drop]])
    # now those should be actual interesting descriptors
    all_descriptors = sorted(set(df.columns) - (required | {"instrum_str", "seed", "pseudotime"}))
    print(f"Descriptors: {all_descriptors}")
    print("# Fight plots")
    #
    # fight plot
    # choice of the combination variables to fix
    fight_descriptors = all_descriptors + ["budget"]  # budget can be used as a descriptor for fight plots
    combinable = [x for x in fight_descriptors if len(df.unique(x)) > 1]  # should be all now
    # We remove descriptors which have only one value for each budget.
    descriptors = []
    for d in all_descriptors:
        acceptable = False
        for b in df.budget.unique():
            if len(df.loc[df["budget"] == b][d].unique()) > 1:
                acceptable = True
                break
        if acceptable:
            descriptors += [d]
    num_rows = 6

    # For the competence map case we must consider pairs of attributes, hence maxcomb_size >= 2.
    # A competence map shows for each value of each of two attributes which algorithm was best.
    if competencemaps:
        max_combsize = max(max_combsize, 2)
    for fixed in list(
        itertools.chain.from_iterable(
            itertools.combinations(combinable, order) for order in range(max_combsize + 1)
        )
    ):
        orders = [len(c) for c in df.unique(fixed)]
        if orders:
            assert min(orders) == max(orders)
            order = min(orders)
        else:
            order = 0
        best_algo: tp.List[tp.List[str]] = []
        if competencemaps and order == 2:  # With order 2 we can create a competence map.
            print("\n#trying to competence-map")
            if all(
                [len(c) > 1 for c in df.unique(fixed)]
            ):  # Let us try if data are adapted to competence maps.
                # This is not always the case, as some attribute1/value1 + attribute2/value2 might be empty
                # (typically when attribute1 and attribute2 are correlated).
                try:
                    xindices = sorted(set(c[0] for c in df.unique(fixed)))
                except TypeError:
                    xindices = list(set(c[0] for c in df.unique(fixed)))
                try:
                    yindices = sorted(set(c[1] for c in df.unique(fixed)))
                except TypeError:
                    yindices = list(set(c[1] for c in df.unique(fixed)))
                for _ in range(len(xindices)):
                    best_algo += [[]]
                for i in range(len(xindices)):
                    for _ in range(len(yindices)):
                        best_algo[i] += ["none"]

        # Let us loop over all combinations of variables.
        for case in df.unique(fixed) if fixed else [()]:
            print("\n# new case #", fixed, case)
            casedf = df.select(**dict(zip(fixed, case)))
            data_df = FightPlotter.winrates_from_selection(
                casedf, fight_descriptors, num_rows=num_rows, num_cols=30
            )
            fplotter = FightPlotter(data_df)
            # Competence maps: we find out the best algorithm for each attribute1=valuei/attribute2=valuej.
            if order == 2 and competencemaps and best_algo:
                print("\n#storing data for competence-map")
                best_algo[xindices.index(case[0])][yindices.index(case[1])] = fplotter.winrates.index[0]
            # save
            name = "fight_" + ",".join("{}{}".format(x, y) for x, y in zip(fixed, case)) + ".png"
            name = "fight_all.png" if name == "fight_.png" else name
            name = compactize(name)
            fullname = name

            if len(name) > 240:
                hashcode = hashlib.md5(bytes(name, "utf8")).hexdigest()
                name = re.sub(r"\([^()]*\)", "", name)
                mid = 120
                name = name[:mid] + hashcode + name[-mid:]
            fplotter.save(str(output_folder / name), dpi=_DPI)
            # Second version, restricted to cases with all data available.
            data_df = FightPlotter.winrates_from_selection(
                casedf, fight_descriptors, num_rows=num_rows, complete_runs_only=True
            )
            fplotter = FightPlotter(data_df)
            if name == "fight_all.png":
                with open(str(output_folder / name) + ".cp.txt", "w") as f:
                    f.write(fullname)
                    f.write("ranking:\n")
                    global pure_algorithms
                    pure_algorithms = list(data_df.columns[:])
                    for i, algo in enumerate(data_df.columns[:158]):
                        f.write(f"  algo {i}: {algo}\n")
            if name == "fight_all.png":
                fplotter.save(str(output_folder / "fight_all_pure.png"), dpi=_DPI)
            else:
                fplotter.save(str(output_folder / name) + "_pure.png", dpi=_DPI)
            if order == 2 and competencemaps and best_algo:  # With order 2 we can create a competence map.
                print("\n# Competence map")
                name = "competencemap_" + ",".join("{}".format(x) for x in fixed) + ".tex"
                export_table(str(output_folder / name), xindices, yindices, best_algo)
                print("Competence map data:", fixed, case, best_algo)

    plt.close("all")
    # xp plots: for each experimental setup, we plot curves with budget in x-axis.
    # plot mean loss / budget for each optimizer for 1 context
    print("# Xp plots")
    name_style = NameStyle()  # keep the same style for each algorithm
    cases = df.unique(descriptors)
    if not cases:
        cases = [()]
    # Average normalized plot with everything.
    out_filepath = output_folder / "xpresults_all.png"
    data = XpPlotter.make_data(df, normalized_loss=True)
    xpplotter = XpPlotter(
        data, title=os.path.basename(output_folder), name_style=name_style, xaxis=xpaxis, pure_only=True
    )
    xpplotter.save(out_filepath)
    # Now one xp plot per case.
    for case in cases:
        subdf = df.select_and_drop(**dict(zip(descriptors, case)))
        description = ",".join("{}:{}".format(x, y) for x, y in zip(descriptors, case))
        full_description = description
        description = compactize(description)
        if len(description) > 280:
            hash_ = hashlib.md5(bytes(description, "utf8")).hexdigest()
            description = description[:140] + hash_ + description[-140:]
        out_filepath = output_folder / "xpresults{}{}.png".format(
            "_" if description else "", description.replace(":", "")
        )
        txt_out_filepath = output_folder / "xpresults{}{}.leaderboard.txt".format(
            "_" if description else "", description.replace(":", "")
        )
        data = XpPlotter.make_data(subdf)
        try:
            xpplotter = XpPlotter(data, title=description, name_style=name_style, xaxis=xpaxis)
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(f"Bypassing error in xpplotter:\n{e}", RuntimeWarning)
        else:
            xpplotter.save(out_filepath)
            xpplotter.save_txt(txt_out_filepath, data, full_description)
    plt.close("all")


def gp_sota() -> tp.Dict[str, tp.Tuple[float, float]]:
    gp = {}
    gp["CartPole-v1"] = (-500.0, 100000.0)
    gp["Acrobot-v1"] = (83.17, 200000.0)
    gp["MountainCarContinuous-v0"] = (-99.31, 900000.0)
    gp["Pendulum-v0"] = (154.36, 1100000.0)
    gp["InvertedPendulumSwingupBulletEnv-v0"] = (-893.35, 400000.0)
    gp["BipedalWalker-v3"] = (-268.85, 1100000.0)
    gp["BipedalWalkerHardcore-v3"] = (-9.25, 1100000.0)
    gp["HopperBulletEnv-v0"] = (-999.19, 1000000.0)
    gp["InvertedDoublePendulumBulletEnv-v0"] = (-9092.17, 300000.0)
    gp["LunarLanderContinuous-v2"] = (-287.58, 1000000.0)
    return gp


class LegendInfo(tp.NamedTuple):
    """Handle for information used to create a legend."""

    x: float
    y: float
    line: tp.Any
    text: str


class XpPlotter:
    """Creates a xp result plot out of the given dataframe: regret with respect to budget for
    each optimizer after averaging on all experiments (it is good practice to use a df
    which is filtered out for one set of input parameters)

    Parameters
    ----------
    optim_vals: dict
        output of the make_data static method, containing all information necessary for plotting
    title: str
        title of the plot
    name_style: dict
        a dict or dict-like object providing a line style for each optimizer name.
        (can be helpful for consistency across plots)
    """

    def __init__(
        self,
        optim_vals: tp.Dict[str, tp.Dict[str, np.ndarray]],
        title: str,
        name_style: tp.Optional[tp.Dict[str, tp.Any]] = None,
        xaxis: str = "budget",
        pure_only: bool = False,
    ) -> None:
        if name_style is None:
            name_style = NameStyle()
        upperbound = max(
            np.max(vals["loss"]) for vals in optim_vals.values() if np.max(vals["loss"]) < np.inf
        )
        for optim, vals in optim_vals.items():
            if optim.lower() in ["stupid", "idiot"] or optim in ["Zero", "StupidRandom"]:
                upperbound = min(upperbound, np.max(vals["loss"]))
        # plot from best to worst
        lowerbound = np.inf
        sorted_optimizers = sorted(optim_vals, key=lambda x: optim_vals[x]["loss"][-1], reverse=True)
        if pure_only:
            assert len(pure_algorithms) > 0
            # print(sorted_optimizers, " merged with ", pure_algorithms)
            sorted_optimizers = [
                o for o in sorted_optimizers if o + " " in [p[: (len(o) + 1)] for p in pure_algorithms]
            ]
            # print("Leads to ", sorted_optimizers)
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        # use log plot? yes, if no negative value
        logplot = not any(
            x <= 0 or x > 10**8 for ov in optim_vals.values() for x in ov["loss"]
        )  # if x < np.inf)
        if logplot:
            self._ax.set_yscale("log")
            for ov in optim_vals.values():
                if ov["loss"].size:
                    ov["loss"] = np.maximum(1e-30, ov["loss"])
        # other setups
        self._ax.autoscale(enable=False)
        self._ax.set_xscale("log")
        self._ax.set_xlabel(xaxis)
        self._ax.set_ylabel("loss")
        self._ax.grid(True, which="both")
        self._overlays: tp.List[tp.Any] = []
        legend_infos: tp.List[LegendInfo] = []
        title_addendum = ""
        for optim_name in (
            sorted_optimizers[:1] + sorted_optimizers[-12:]
            if len(sorted_optimizers) > 13
            else sorted_optimizers
        ):
            vals = optim_vals[optim_name]
            indices = np.where(vals["num_eval"] > 0)
            lowerbound = min(lowerbound, np.min(vals["loss"]))
            # We here add some state of the art results.
            # This adds a cross on figures, x-axis = budget and y-axis = loss.
            for sota_name, sota in [("GP", gp_sota())]:
                for k in sota.keys():
                    if k in title:
                        th = sota[k][0]  # loss of proposed solution.
                        cost = sota[k][1]  # Computational cost for the proposed result.
                        title_addendum = f"({sota_name}:{th})"
                        lowerbound = min(lowerbound, th, 0.9 * th, 1.1 * th)
                        plt.plot(  # Horizontal line at the obtained GP cost.
                            vals[xaxis][indices],
                            th + 0 * vals["loss"][indices],
                            name_style[optim_name],
                            label="gp",
                        )
                        plt.plot(  # Vertical line, showing the budget of the GP solution.
                            [cost] * 3,
                            [
                                min(vals["loss"][indices]),
                                sum(vals["loss"][indices]) / len(indices),
                                max(vals["loss"][indices]),
                            ],
                            name_style[optim_name],
                            label="gp",
                        )
            line = plt.plot(vals[xaxis], vals["loss"], name_style[optim_name], label=optim_name)
            # confidence lines
            for conf in self._get_confidence_arrays(vals, log=logplot):
                plt.plot(vals[xaxis], conf, name_style[optim_name], label=optim_name, alpha=0.1)
            text = "{} ({:.3g} <{:.3g}>)".format(
                optim_name,
                vals["loss"][-1],
                vals["loss"][-2] if len(vals["loss"]) > 2 else float("nan"),
            )
            if vals[xaxis].size:
                legend_infos.append(LegendInfo(vals[xaxis][-1], vals["loss"][-1], line, text))
        if not (np.isnan(upperbound) or np.isinf(upperbound)):
            upperbound_up = upperbound
            if not (np.isnan(lowerbound) or np.isinf(lowerbound)):
                self._ax.set_ylim(bottom=lowerbound)
                upperbound_up += 0.02 * (upperbound - lowerbound)
                if logplot:
                    upperbound_up = 10 ** (
                        np.log10(upperbound) + 0.02 * (np.log10(upperbound) - np.log10(lowerbound))
                    )
            self._ax.set_ylim(top=upperbound_up)
        all_x = [v for vals in optim_vals.values() for v in vals[xaxis]]
        self._ax.set_xlim([min(all_x), max(all_x)])
        self.add_legends(legend_infos)
        # global info
        if "tmp" not in title:
            self._ax.set_title(split_long_title(title + title_addendum))
        self._ax.tick_params(axis="both", which="both")
        # self._fig.tight_layout()

    @staticmethod
    def _get_confidence_arrays(
        vals: tp.Dict[str, np.ndarray], log: bool = False
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        loss = vals["loss"]
        conf = vals["loss_std"] / np.sqrt(vals["num_eval"] - 1)
        if not log:
            return loss - conf, loss + conf
        lloss = np.log10(loss)
        lstd = 0.434 * conf / loss
        return tuple(10 ** (lloss + x) for x in [-lstd, lstd])  # type: ignore

    def add_legends(self, legend_infos: tp.List[LegendInfo]) -> None:
        """Adds the legends"""
        ax = self._ax
        trans = ax.transScale + ax.transLimits
        fontsize = 10.0
        display_y = (ax.transAxes.transform((1, 1)) - ax.transAxes.transform((0, 0)))[1]  # height in points
        shift = (2.0 + fontsize) / display_y
        legend_infos = legend_infos[::-1]  # revert order for use in compute_best_placements
        values = [float(np.clip(trans.transform((0, i.y))[1], -0.01, 1.01)) for i in legend_infos]
        placements = compute_best_placements(values, min_diff=shift)
        for placement, info in zip(placements, legend_infos):
            self._overlays.append(
                Legend(
                    ax,
                    info.line,
                    [info.text],
                    loc="center left",
                    bbox_to_anchor=(1, placement),
                    frameon=False,
                    fontsize=fontsize,
                )
            )
            ax.add_artist(self._overlays[-1])

    @staticmethod
    def make_data(df: pd.DataFrame, normalized_loss: bool = False) -> tp.Dict[str, tp.Dict[str, np.ndarray]]:
        """Process raw xp data and process it to extract relevant information for xp plots:
        regret with respect to budget for each optimizer after averaging on all experiments (it is good practice to use a df
        which is filtered out for one set of input parameters)

        Parameters
        ----------
        df: pd.DataFrame
            run data
        normalized_loss: bool
            whether we should normalize each data (for each budget and run) between 0 and 1. Convenient when we consider
            averages over several distinct functions that can have very different ranges - then we return data which are rescaled to [0,1].
            Warning: then even if algorithms converge (i.e. tend to minimize), the value can increase, because the normalization
            is done separately for each budget.
        """
        if normalized_loss:
            descriptors = sorted(
                set(df.columns)
                - {
                    "pseudotime",
                    "time",
                    "budget",
                    "elapsed_time",
                    "elapsed_budget",
                    "loss",
                    "optimizer_name",
                    "seed",
                }
            )
            df = normalized_losses(df, descriptors=descriptors)
        df = utils.Selector(
            df.loc[
                :,
                ["optimizer_name", "budget", "loss"] + (["pseudotime"] if "pseudotime" in df.columns else []),
            ]
        )
        groupeddf = df.groupby(["optimizer_name", "budget"])
        means = groupeddf.mean()
        stds = groupeddf.std()
        optim_vals: tp.Dict[str, tp.Dict[str, np.ndarray]] = {}
        # extract name and coordinates
        for optim in df.unique("optimizer_name"):
            optim_vals[optim] = {}
            optim_vals[optim]["budget"] = np.array(means.loc[optim, :].index)
            optim_vals[optim]["loss"] = np.array(means.loc[optim, "loss"])
            optim_vals[optim]["loss_std"] = np.array(stds.loc[optim, "loss"])
            num_eval = np.array(groupeddf.count().loc[optim, "loss"])
            optim_vals[optim]["num_eval"] = num_eval
            if "pseudotime" in means.columns:
                optim_vals[optim]["pseudotime"] = np.array(means.loc[optim, "pseudotime"])
        return optim_vals

    @staticmethod
    def save_txt(
        output_filepath: tp.PathLike, optim_vals: tp.Dict[str, tp.Dict[str, np.ndarray]], addendum: str = ""
    ) -> None:
        """Saves a list of best performances.

        output_filepath: Path or str
            path where the figure must be saved
        optim_vals: dict
            dict of losses obtained by a given optimizer.
        """
        best_performance: tp.Dict[int, tp.Any] = defaultdict(lambda: (float("inf"), "none"))
        for optim in optim_vals.keys():
            for i, l in zip(optim_vals[optim]["budget"], optim_vals[optim]["loss"]):
                if l < best_performance[i][0]:
                    best_performance[i] = (l, optim)

        with open(output_filepath, "w") as f:
            f.write(addendum)
            f.write("Best performance:\n")
            for i in best_performance.keys():
                f.write(
                    f"  budget {i}: {best_performance[i][0]} ({best_performance[i][1]}) ({output_filepath})\n"
                )

    def save(self, output_filepath: tp.PathLike) -> None:
        """Saves the xp plot

        Parameters
        ----------
        output_filepath: Path or str
            path where the figure must be saved
        """
        try:  # Let us catch errors due to too many DPIs.
            self._fig.savefig(
                str(output_filepath), bbox_extra_artists=self._overlays, bbox_inches="tight", dpi=_DPI
            )
        except ValueError as v:
            print(f"We catch {v} which means that image = too big.")
            self._fig.savefig(
                str(output_filepath), bbox_extra_artists=self._overlays, bbox_inches="tight", dpi=_DPI / 5
            )

    def __del__(self) -> None:
        plt.close(self._fig)


def split_long_title(title: str) -> str:
    """Splits a long title around the middle comma"""
    if len(title) <= 60:
        return title
    comma_indices = np.where(np.array(list(title)) == ",")[0]
    if not comma_indices.size:
        return title
    best_index = comma_indices[np.argmin(abs(comma_indices - len(title) // 2))]
    title = title[: (best_index + 1)] + "\n" + title[(best_index + 1) :]
    return title


# @contextlib.contextmanager
# def xticks_on_top() -> tp.Iterator[None]:
#     values_for_top = {'xtick.bottom': False, 'xtick.labelbottom': False,
#                       'xtick.top': True, 'xtick.labeltop': True}
#     defaults = {x: plt.rcParams[x] for x in values_for_top if x in plt.rcParams}
#     plt.rcParams.update(values_for_top)
#     yield
#     plt.rcParams.update(defaults)


class FightPlotter:
    """Creates a fight plot out of the given dataframe, by iterating over all cases with fixed category variables.

    Parameters
    ----------
    winrates_df: pd.DataFrame
        winrate data as a dataframe
    """

    def __init__(self, winrates_df: pd.DataFrame) -> None:
        # make plot
        self.winrates = winrates_df
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._cax = self._ax.imshow(
            100 * np.array(self.winrates), cmap=cm.seismic, interpolation="none", vmin=0, vmax=100
        )
        x_names = self.winrates.columns
        self._ax.set_xticks(list(range(len(x_names))))
        self._ax.set_xticklabels(x_names, rotation=45, ha="right", fontsize=7)
        y_names = self.winrates.index
        self._ax.set_yticks(list(range(len(y_names))))
        self._ax.set_yticklabels(y_names, rotation=45, fontsize=7)
        divider = make_axes_locatable(self._ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # self._fig.colorbar(im, cax=cax)
        self._fig.colorbar(self._cax, cax=cax)  # , orientation='horizontal')
        plt.tight_layout()

    @staticmethod
    def winrates_from_selection(
        df: utils.Selector,
        categories: tp.List[str],
        num_rows: int = 5,
        num_cols: int = 30,
        complete_runs_only: bool = False,
    ) -> pd.DataFrame:
        """Creates a fight plot win rate data out of the given run dataframe,
        by iterating over all cases with fixed category variables.

        Parameters
        ----------
        df: pd.DataFrame
            run data
        categories: list
            List of variables to fix for obtaining similar run conditions
        num_rows: int
            number of rows to plot (best algorithms)
        complete_runs_only: bool
            if we want a plot with only algorithms which have run on all settings
        """
        all_optimizers = list(df.unique("optimizer_name"))  # optimizers for which no run exists are not shown
        num_rows = min(num_rows, len(all_optimizers))
        # iterate on all sub cases
        victories, total = aggregate_winners(df, categories, all_optimizers)
        if complete_runs_only:
            max_num = max([int(2 * victories.loc[n, n]) for n in all_optimizers])
            new_all_optimizers = [n for n in all_optimizers if int(2 * victories.loc[n, n]) == max_num]
            if len(new_all_optimizers) > 0:
                df = df[df["optimizer_name"].isin(new_all_optimizers)]
                victories, total = aggregate_winners(df, categories, new_all_optimizers)
        # subcases = df.unique(categories)
        # for k, subcase in enumerate(subcases):  # TODO linearize this (precompute all subcases)? requires memory
        #     # print(subcase)
        #     subdf = df.select(**dict(zip(categories, subcase)))
        #     victories += _make_winners_df(subdf, all_optimizers)
        #     if k > 1000:
        #         break
        winrates = _make_sorted_winrates_df(victories)
        mean_win = winrates.mean(axis=1)
        winrates.fillna(0.5)  # unplayed
        sorted_names = winrates.index
        # number of subcases actually computed is twice self-victories
        sorted_names = ["{} ({}/{})".format(n, int(2 * victories.loc[n, n]), total) for n in sorted_names]
        num_names = len(sorted_names)
        sorted_names = [sorted_names[i] for i in range(min(num_cols, num_names))]
        data = np.array(winrates.iloc[:num_rows, : len(sorted_names)])
        # pylint: disable=anomalous-backslash-in-string
        best_names = [
            (
                f"{name} ({i+1}/{num_names}:{100 * val:2.1f}% +- {25 * np.sqrt(val*(1-val)/int(2 * victories.loc[name, name])):2.1f})"
            ).replace("Search", "")
            for i, (name, val) in enumerate(zip(mean_win.index[:num_rows], mean_win))
        ]
        return pd.DataFrame(index=best_names, columns=sorted_names, data=data)

    def save(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Shortcut to the figure savefig method"""
        self._fig.savefig(*args, **kwargs)

    def __del__(self) -> None:
        plt.close(self._fig)


# %% positionning legends


class LegendGroup:
    """Class used to compute legend best placements.
    Each group contains at least one legend, and has a position and span (with bounds). LegendGroup are then
    responsible for providing each of its legends' position (non-overlapping)


    Parameters
    ----------
    indices: List[int]
        identifying index of each of the legends
    init_position: List[float]
        best position for each of the legends (if there was no overlapping)
    min_diff: float
        minimal distance between two legends so that they do not overlap
    """

    def __init__(self, indices: tp.List[int], init_positions: tp.List[float], min_diff: float):
        assert all(x2 - x1 == 1 for x2, x1 in zip(indices[1:], indices[:-1]))
        assert all(v2 >= v1 for v2, v1 in zip(init_positions[1:], init_positions[:-1]))
        assert len(indices) == len(init_positions)
        self.indices = indices
        self.init_positions = init_positions
        self.min_diff = min_diff
        self.position = float(np.mean(init_positions))

    def combine_with(self, other: "LegendGroup") -> "LegendGroup":
        assert self.min_diff == other.min_diff
        return LegendGroup(
            self.indices + other.indices, self.init_positions + other.init_positions, self.min_diff
        )

    def get_positions(self) -> tp.List[float]:
        first_position = self.bounds[0] + self.min_diff / 2.0
        return [first_position + k * self.min_diff for k in range(len(self.indices))]

    @property
    def bounds(self) -> tp.Tuple[float, float]:
        half_span = len(self.indices) * self.min_diff / 2.0
        return (self.position - half_span, self.position + half_span)

    def __repr__(self) -> str:
        return f"LegendGroup({self.indices}, {self.init_positions}, {self.min_diff})"


def compute_best_placements(positions: tp.List[float], min_diff: float) -> tp.List[float]:
    """Provides a list of new positions from a list of initial position, with a minimal
    distance between each position.

    Parameters
    ----------
    positions: List[float]
        best positions if minimal distance were 0.
    min_diff: float
        minimal distance allowed between two positions

    Returns
    -------
    new_positions: List[float]
        positions after taking into account the minimal distance constraint

    Note
    ----
    This function is probably not optimal, but seems a very good heuristic
    """
    assert all(v2 >= v1 for v2, v1 in zip(positions[1:], positions[:-1]))
    groups = [LegendGroup([k], [pos], min_diff) for k, pos in enumerate(positions)]
    new_groups: tp.List[LegendGroup] = []
    ready = False
    while not ready:
        ready = True
        for k in range(len(groups)):  # pylint: disable=consider-using-enumerate
            if k < len(groups) - 1 and groups[k + 1].bounds[0] < groups[k].bounds[1]:
                # groups are overlapping: create a new combined group
                # which will provide new non-overlapping positions around the mean of initial positions
                new_groups.append(groups[k].combine_with(groups[k + 1]))
                # copy the rest of the groups and start over from the first group
                new_groups.extend(groups[k + 2 :])
                groups = new_groups
                new_groups = []
                ready = False
                break
            new_groups.append(groups[k])
    new_positions = np.array(positions, copy=True)
    for group in groups:
        new_positions[group.indices] = group.get_positions()
    return new_positions.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create plots from an experiment data file")
    parser.add_argument("filepath", type=str, help="filepath containing the experiment data")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the CSV file (default: a folder <filename>_plots next to the data file.",
    )
    parser.add_argument(
        "--max_combsize",
        type=int,
        default=0,
        help="maximum number of parameters to fix (combinations) when creating experiment plots",
    )
    parser.add_argument(
        "--pseudotime",
        nargs="?",
        default=False,
        const=True,
        help="Plots with respect to pseudotime instead of budget",
    )
    parser.add_argument(
        "--competencemaps", type=bool, default=False, help="whether we should export only competence maps"
    )
    parser.add_argument(
        "--merge-parametrization",
        action="store_true",
        help="if present, parametrization is merge into the optimizer name",
    )
    parser.add_argument(
        "--remove-suffix",
        action="store_true",
        help="if present, remove numerical suffixes in fight plots",
    )
    parser.add_argument(
        "--merge-pattern",
        type=str,
        default="",
        help="if present, optimizer name is updated according to the pattern as "
        f"an f-string. --merge-parametrization is equivalent to using --merge-pattern with {_PARAM_MERGE_PATTERN!r}",
    )
    args = parser.parse_args()
    exp_df = merge_optimizer_name_pattern(
        utils.Selector.read_csv(args.filepath),
        args.merge_pattern,
        args.merge_parametrization,
        args.remove_suffix,
    )
    # merging names
    #
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(args.filepath).with_suffix("")) + "_plots"
    create_plots(
        exp_df,
        output_folder=output_dir,
        max_combsize=args.max_combsize if not args.competencemaps else 2,
        xpaxis="pseudotime" if args.pseudotime else "budget",
        competencemaps=args.competencemaps,
    )


if __name__ == "__main__":
    main()
