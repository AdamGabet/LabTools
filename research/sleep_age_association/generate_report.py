import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


def generate_report():
    study_name = "sleep_age_association"
    fig_path = f"research/{study_name}/figures/sleep_age_association.png"
    pdf_path = f"research/{study_name}/report.pdf"

    if not os.path.exists(fig_path):
        print(f"Error: Figure not found at {fig_path}")
        return

    with PdfPages(pdf_path) as pdf:
        # Create a page for the main figure
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread(fig_path)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis("off")
        plt.title("Figure 2 Recreation: Sleep Features vs Age", fontsize=16, pad=-20)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report saved to {pdf_path}")


if __name__ == "__main__":
    generate_report()
