"""
Analyze learning curves: test set predictions of models trained on different
training set sizes
"""

from typing import List, Optional, Literal
import numpy as np
from matplotlib.axes import Axes


def subsample_errors(errors: np.ndarray, n_samples: int):
    """
    Subsample force errors for faster violin plot generation

    Parameters
    ----------
    errors : np.ndarray
        Array of errors (energy or force components), shape (n,)
    n_samples : int
        Number of samples to subsample

    Returns
    -------
    np.ndarray
        Subsampled array of errors, including min and max values
    """
    max_outlier = [np.max(errors)]
    min_outlier = [np.min(errors)]
    random_sample = np.random.choice(errors, size=n_samples, replace=True)
    return np.concatenate([max_outlier, min_outlier, random_sample])


class LearningCurve:
    """
    Tool to plot learning curves for machine learning interatomic potentials. In this context
    a learning curve plot shows the test error of the model against the training set size.
    For all data points the model should be fully trained. (These learning curves are not to be
    confused with test/train error vs. #epochs curves obtained from training one model).

    RMSE (test)
    ^  |  *
    |  |   *
       |     *
       |        *   *   *   *
       |______________________
                        --> dataset size

    Parameters
    ----------
    dft_forces : np.ndarray
        DFT forces for the test set, shape (n_configs, n_atoms, 3)
    dft_energy : np.ndarray
        DFT energy for the test set, shape (n_configs,)
    """

    def __init__(self, dft_forces: np.ndarray, dft_energy: np.ndarray):
        self.n_atoms = dft_forces.shape[1]
        # TODO: generalize for validation sets with different number of atoms per frame
        # (accept list of np.ndarray for each config)
        self.dft_forces = dft_forces
        self.dft_energy = dft_energy
        self.training_set_sizes = []
        self.errors = {
            "force_component": [],
            "energy_per_atom": [],
        }

    def add_training_set(
        self,
        n_training_samples: int,
        ml_forces: List[np.ndarray],
        ml_energy: List[np.ndarray],
    ):
        """
        Add a training set to the learning curve plot. One can specify a list of multiple
        models, for example multiple models trained with different random seeds, or trained
        on different subsets of the training data but evaluated on the same test set.

        Parameters
        ----------
        n_training_samples : int
            Number of training samples used
        ml_forces : List[np.ndarray]
            List of ML forces for each model, shape (n_configs, n_atoms, 3).
        ml_energy : List[np.ndarray]
            List of ML energies for each model, shape (n_configs,)
        """
        assert len(ml_forces) == len(ml_energy)
        assert isinstance(ml_forces, list)
        assert isinstance(ml_energy, list)

        self.training_set_sizes.append(n_training_samples)
        force_component_errors = []
        energy_per_atom_errors = []
        for ml_f, ml_e in zip(ml_forces, ml_energy):
            f_err = np.array(ml_f - self.dft_forces).flatten() * 1e3  # meV/A
            e_err = np.array(ml_e - self.dft_energy) / self.n_atoms * 1e3  # meV/atom
            force_component_errors.append(f_err)
            energy_per_atom_errors.append(e_err)

        self.errors["force_component"].append(force_component_errors)
        self.errors["energy_per_atom"].append(energy_per_atom_errors)

    def make_violin(
        self,
        axes: Axes,
        error_type: Literal["force_component", "energy_per_atom"] = "force_component",
        n_subsampling: Optional[int] = None,
        **kwargs,
    ):
        """
        Make a violin plot with matplotlib.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Matplotlib axes object where the violin plot will be drawn.
        error_type : {'force_component', 'energy_per_atom'}, optional
            Type of error to plot. Options are "force_component" or "energy_per_atom".
            Default is "force_component".
        n_subsampling : int, optional
            Number of samples to subsample for faster plotting. If None, no subsampling is
            performed. Default is None.

        Other Parameters
        ----------------
        side : str, optional
            Side of the violin plot to display. Options are "both", "left", or "right".
            Default is "both".
        face_color : str, optional
            Color of the violin plot. Default is "blue".
        box_percentiles : tuple, optional
            Percentiles to use for the box plot overlay. Default is (0.05, 99.95).
        x_tick_labels : list, optional
            Labels for the x-ticks. If None, integer indices will be used (1, 2, 3...).
            Default is None.
        """
        # Extract optional arguments
        box_percentiles = kwargs.get("box_percentiles", (0.05, 99.95))

        # Find errors for each training set
        error_list = []
        for set_errors in self.errors[error_type]:
            set_errors = np.array(set_errors).flatten()

            if n_subsampling:
                set_errors = subsample_errors(set_errors, n_subsampling)
            error_list.append(set_errors)

        positions = np.arange(len(self.training_set_sizes)) + 1
        violin = axes.violinplot(
            error_list,
            positions=positions,
            widths=0.7,
            showextrema=False,
            side=kwargs.get("side", "both"),
        )

        # Set color for KDE distribution
        for pc in violin["bodies"]:
            pc.set_facecolor(kwargs.get("face_color", "blue"))
            pc.set_edgecolor("black")
            pc.set_alpha(1)

        # Add percentile box
        if box_percentiles is not None:
            quartile1, _, quartile3 = np.percentile(
                error_list, [box_percentiles[0], 50, box_percentiles[-1]], axis=1
            )
            axes.vlines(positions, quartile1, quartile3, color="k", linestyle="-", lw=5)

        # Add error bars for min and max
        min_err = np.min(error_list, axis=1)
        max_err = np.max(error_list, axis=1)
        axes.vlines(positions, min_err, max_err, color="k", linestyle="-", lw=1)

        if kwargs.get("x_tick_labels", None):
            axes.set_xticks(positions, kwargs.get("x_tick_labels", None))
        else:
            axes.set_xticks(positions)

        axes.set_xlabel("Training set")

        if error_type == "force_component":
            axes.set_ylabel(r"Error / meV/$\mathrm{\AA}$")
        elif error_type == "energy_per_atom":
            axes.set_ylabel("Error / meV")

        return violin

    def make_learningcurve(
        self,
        axes: Axes,
        legend=True,
        error_type: Literal["force_component", "energy_per_atom="] = "force_component",
        **kwargs,
    ):
        """
        Plot test RMSE obtained on each training set against training set size.

        Parameters
        ----------
        axes : Axes
            The matplotlib axes to plot on.
        legend : bool, optional
            Whether to display the legend (default is True).
        error_type : {'force_component', 'energy_per_atom='}, optional
            The type of error to plot (default is 'force_component').
        **kwargs : dict
            Additional keyword arguments passed to the plot function.

        Returns
        -------
        eb : ErrorbarContainer
            The error bar container object.
        """
        mean_rmse = []
        min_rmse = []
        max_rmse = []

        for set_errors in self.errors[error_type]:
            rmse = [np.sqrt(np.mean(model_err**2)) for model_err in set_errors]
            mean_rmse.append(np.mean(rmse))
            min_rmse.append(np.min(rmse))
            max_rmse.append(np.max(rmse))

        mean_rmse = np.array(mean_rmse)
        min_rmse = np.array(min_rmse)
        max_rmse = np.array(max_rmse)

        # Plot points at mean RMSE
        eb = axes.plot(
            self.training_set_sizes,
            mean_rmse,
            **kwargs,
        )

        # Make error bars for min/max
        eb = axes.errorbar(
            self.training_set_sizes,
            mean_rmse,
            yerr=[mean_rmse - min_rmse, max_rmse - mean_rmse],
            fmt="none",  # Marker style
            ecolor="black",  # Color of the error bars
            capsize=5,  # Length of the error bar caps
            zorder=100,
        )

        axes.set_xlabel("Training set size")
        if error_type == "force_component":
            axes.set_ylabel(r"RMSE / meV/$\mathrm{\AA}$")
        elif error_type == "energy_per_atom":
            axes.set_ylabel("RMSE / meV")
        if legend:
            axes.legend(frameon=False)

        return eb
