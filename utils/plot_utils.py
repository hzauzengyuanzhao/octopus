import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Arial']

class MatrixPlot:
    def __init__(self, output_path, image, prefix, celltype, chr_name, start_pos):
        self.output_path = output_path,
        self.prefix = prefix
        self.celltype = celltype
        self.chr_name = chr_name
        self.start_pos = start_pos

        self.create_save_path(output_path, celltype, prefix)
        self.image = self.preprocess_image(image)

    def get_colormap(self):
        colors = [
            (1, 1, 1),
            (1, 0.9, 0.9),
            (0.9, 0.6, 0.6),
            (0.8, 0.3, 0.3),
            (0.7, 0.1, 0.1),
            (0.5, 0, 0)
        ]
        return LinearSegmentedColormap.from_list("enhanced_red", colors, N=256)

    def create_save_path(self, output_path, celltype, prefix):
        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok = True)
        os.makedirs(f'{self.save_path}/npy', exist_ok = True)

    def preprocess_image(self, image):
        return image

    def plot(self, vmin = 0, vmax = 5, contrast_boost=1.5):
        fig, ax = plt.subplots(figsize = (5, 5))
        color_map = self.get_colormap()
        cax = ax.imshow(self.image, cmap=color_map,aspect='equal', vmin=vmin, vmax=vmax,interpolation='nearest')
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([vmin, vmax])
        cbar.set_label('')
        self.reformat_ticks(plt)
        return

    def reformat_ticks(self, plt):
        # Rescale tick labels
        current_ticks = np.arange(0, 250, 50) / 0.8192
        plt.xticks(current_ticks, self.rescale_coordinates(current_ticks, self.start_pos))
        plt.yticks(current_ticks, self.rescale_coordinates(current_ticks, self.start_pos))
        # Format labels
        plt.ylabel('Genomic position (Mb)')
        plt.xlabel(f'{self.chr_name}: {self.start_pos} - {self.start_pos + 2097152} ')
        self.save_data(plt)

    def rescale_coordinates(self, coords, zero_position):
        scaling_ratio = 8192
        replaced_coords = coords * scaling_ratio + zero_position
        coords_mb = replaced_coords / 1000000
        str_list = [f'{item:.2f}' for item in coords_mb]
        return str_list

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}.pdf',dpi=300, bbox_inches = 'tight',format='pdf')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}', self.image)


class MatrixPlotDeletion(MatrixPlot):
    def __init__(self, output_path, image, prefix, celltype, chr_name, start_pos, deletion_start, deletion_width, padding_type, show_deletion_line = False):
        super().__init__(output_path, image, prefix, celltype, chr_name, start_pos)
        self.deletion_start = deletion_start
        self.deletion_width = deletion_width
        self.show_deletion_line = show_deletion_line
        self.padding_type = padding_type

    def reformat_ticks(self, plt):
        # Rescale tick labels
        breakpoint_start = (self.deletion_start - self.start_pos) / 10000
        breakpoint_end = (self.deletion_start - self.start_pos + self.deletion_width) / 10000
        # Used for generating ticks until the end of the window
        total_window_size = (self.deletion_width + 2097152 ) / 10000
        # Generate ticks before and after breakpoint
        before_ticks = np.arange(0, breakpoint_start - 50, 50) / 0.8192
        after_ticks = (np.arange((breakpoint_end // 50 + 2) * 50, total_window_size, 50) - self.deletion_width / 10000) / 0.8192
        breakpoint_locus = breakpoint_start / 0.8192
        # Actual coordinates for each tick
        current_ticks = np.append(before_ticks, after_ticks)
        current_ticks = np.append(current_ticks, breakpoint_start / 0.8192)
        # Genomic coordinates used for display location after deletion
        display_ticks = np.append(before_ticks, after_ticks + self.deletion_width / 10000 / 0.8192)
        display_ticks = np.append(display_ticks, breakpoint_start / 0.8192)
        if self.show_deletion_line:
            plt.axline((breakpoint_locus, 0), (breakpoint_locus, 209), c = 'black', alpha = 0.5)
            plt.axline((0, breakpoint_locus), (209, breakpoint_locus), c = 'black', alpha = 0.5)
        # Generate tick label text
        ticks_label = self.rescale_coordinates(display_ticks, self.start_pos)
        plt.yticks(current_ticks, ticks_label)
        ticks_label[-1] = f"{(self.deletion_start / 1000000):.2f}({(self.deletion_start + self.deletion_width) / 1000000:.2f})"
        plt.xticks(current_ticks, ticks_label)
        # Format labels
        plt.ylabel('Genomic position (Mb)')
        end_pos = self.start_pos + 2097152 + self.deletion_width
        plt.xlabel(f'{self.chr_name}: {self.start_pos} - {self.deletion_start} and {self.deletion_start + self.deletion_width} - {end_pos} ')
        self.save_data(plt)

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}.pdf', dpi=300, bbox_inches = 'tight',format='pdf')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}', self.image)

class MatrixPlotPointScreen(MatrixPlotDeletion):
    def plot(self, vmin = -1, vmax = 1):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize = (5, 5))
        cax = ax.imshow(self.image, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([vmin, vmax])
        self.reformat_ticks(plt)
        return

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}_diff.pdf',dpi=300, bbox_inches = 'tight',format='pdf')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}_diff', self.image)