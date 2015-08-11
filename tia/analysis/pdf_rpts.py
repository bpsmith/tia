import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle, TA_CENTER

import tia.rlab as rlab
from tia.analysis.model.port import PortfolioSummary
import tia.analysis.perf as perf
from tia.util.fmt import new_datetime_formatter
from tia.util.mplot import AxesFormat, FigureHelper
from tia.analysis.util import insert_level


class _Result(object):
    def __init__(self, port, sid, desc):
        self.port = port
        self.buyhold = port.buy_and_hold()
        self.sid = sid
        self.desc = desc


class ShortTermReport(object):
    """Used when showing 2 years or less worth of portfolio returns"""

    def __init__(self, path, title, author=None, table_style=None):
        self.path = path
        self.pdf = None
        self.results = []
        self.title = title
        self.author = author
        self.table_style = table_style or rlab.Style.Black
        self.figures = FigureHelper(dpi=300)
        self.long_color = plt.rcParams['axes.color_cycle'][0]
        self.short_color = plt.rcParams['axes.color_cycle'][1]

    def add_port(self, port, sid, desc):
        self.results.append(_Result(port, sid, desc))

    def create_ax(self, figsize=(5, 3)):
        return plt.subplots(1, 1, figsize=figsize)

    def define_portfolio_summary_template(self):
        t = rlab.GridTemplate('portfolio', 100, 100)
        r1 = slice(5, 30)
        r2 = slice(30, 55)
        r3 = slice(55, 100)
        c1 = slice(1, 33)
        c2 = slice(33, 65)
        c3 = slice(66, 98)
        t.define_frames({'F1': t[r1, c1], 'F2': t[r1, c2], 'F3': t[r1, c3], 'F4': t[r2, c1], 'F5': t[r2, c2],
                         'F6': t[r2, c3], 'F7': t[r3, c1], 'F8': t[r3, 33:], 'HDR': t[:5, :]})
        t.register(self.pdf)

    def define_position_summary_template(self):
        t = rlab.GridTemplate('positions', 100, 100)
        r1 = slice(5, 30)
        r2 = slice(30, 55)
        r3 = slice(55, 100)
        c1 = slice(1, 50)
        c2 = slice(50, 98)

        t.define_frames({'F1': t[r1, c1], 'F2': t[5:55, c2], 'F3': t[r2, c1], 'F5': t[r3, 1:33], 'F6': t[r3, 33:],
                         'HDR': t[:5, :]})
        t.register(self.pdf)

    def define_summary_template(self):
        t = rlab.GridTemplate('summary', 100, 100)
        t.define_frames({'F1': t[1:99, 1:99]})
        t.register(self.pdf)

    def add_summary_page(self):
        """Build a table which is shown on the first page which gives an overview of the portfolios"""
        s = PortfolioSummary()
        s.include_long_short()
        pieces = []
        for r in self.results:
            tmp = s(r.port, PortfolioSummary.analyze_returns)
            tmp['desc'] = r.desc
            tmp['sid'] = r.sid
            tmp = tmp.set_index(['sid', 'desc'], append=1).reorder_levels([2, 1, 0])
            pieces.append(tmp)
        frame = pd.concat(pieces)

        tf = self.pdf.table_formatter(frame)
        tf.apply_basic_style(cmap=self.table_style)
        # [col.guess_format(pcts=1, trunc_dot_zeros=1) for col in tf.cells.iter_cols()]
        tf.cells.match_column_labels(['nmonths', 'cnt', 'win cnt', 'lose cnt', 'dur max']).int_format()
        tf.cells.match_column_labels(['sharpe ann', 'sortino', 'dur avg']).float_format(precision=1)
        tf.cells.match_column_labels(['maxdd dt']).apply_format(new_datetime_formatter('%d-%b-%y'))
        tf.cells.match_column_labels(['cagr', 'mret avg', 'mret std ann', 'ret std', 'mret avg ann', 'maxdd', 'avg dd',
                                      'winpct', 'ret avg', 'ret min', 'ret max']).percent_format()

        self.pdf.build_page('summary', {'F1': tf.build()})

    def title_bar(self, title):
        # Build a title bar for top of page
        w, t, c = '100%', 2, HexColor('#404040')
        title = '<b>{0}</b>'.format(title)
        return [HRFlowable(width=w, thickness=t, color=c, spaceAfter=2, vAlign='MIDDLE', lineCap='square'),
                self.pdf.new_paragraph(title, 'TitleBar'),
                HRFlowable(width=w, thickness=t, color=c, spaceBefore=2, vAlign='MIDDLE', lineCap='square')]

    def run(self):
        cp = rlab.CoverPage(self.title, subtitle2=self.author)
        self.pdf = pdf = rlab.PdfBuilder(self.path, coverpage=cp)
        # Setup stylesheet
        tb = ParagraphStyle('TitleBar', parent=pdf.stylesheet['Normal'], fontName='Helvetica-Bold', fontSize=10,

                    leading=10, alignment=TA_CENTER)
        'TitleBar' not in pdf.stylesheet and pdf.stylesheet.add(tb)
        # define templates
        self.define_portfolio_summary_template()
        self.define_position_summary_template()
        self.define_summary_template()
        # Show the summary page
        self.add_summary_page()
        # Build the portfolio and position details for each result
        for r in self.results:
            self.add_portfolio_page(r)
            self.add_position_page(r)
        pdf.save()

    def add_portfolio_page(self, result):
        def alpha_beta(p, bm):
            model = pd.ols(x=bm.rets, y=p.rets)
            beta = model.beta[0]
            alpha = p.total_ann - beta * bm.total_ann
            s = pd.Series({'alpha': alpha, 'beta': beta})
            return s

        def rs(port1, port2, kind='dly_ret_stats'):
            stats = getattr(port1, kind)
            ab = alpha_beta(stats, getattr(port2, kind))
            tmp = stats.series.append(ab)
            tmp.name = stats.series.name
            return tmp

        def dofmt(t):
            t.apply_basic_style(cmap=self.table_style)
            [row.guess_format(pcts=1, trunc_dot_zeros=1) for row in t.cells.iter_rows()]
            ncols = len(t.formatted_values.columns)
            t.set_col_widths(pcts=[1. / ncols] * ncols)

        def do_rename(df):
            d = {'consecutive_win_cnt_max': 'win_streak', 'consecutive_loss_cnt_max': 'lose_streak'}
            return df.rename(index=lambda c: d.get(c, c))

        # Build the pdf tables
        pdf = self.pdf
        figures = self.figures
        port = result.port
        buyhold = result.buyhold
        sframe = pd.DataFrame([rs(port, buyhold, 'dly_ret_stats'),
                               rs(port, buyhold, 'weekly_ret_stats'),
                               rs(port, buyhold, 'monthly_ret_stats'),
                               rs(port, buyhold, 'quarterly_ret_stats')]).T

        tf = pdf.table_formatter(insert_level(sframe, 'Portfolio', copy=True))
        dofmt(tf)
        stable = tf.build()

        s = PortfolioSummary()
        s.include_long_short().include_win_loss()
        dframe = s(port, PortfolioSummary.analyze_returns).T
        tf = pdf.table_formatter(do_rename(insert_level(dframe.ix['port'], 'Portfolio', copy=True)))
        dofmt(tf)
        dtable = tf.build()

        # Return on $1 image
        f, ax = self.create_ax()
        buyhold.plot_ret_on_dollar('B', label='Buy & Hold', ax=ax)
        port.plot_ret_on_dollar('B', label=result.desc, ax=ax, color='k')
        ax.legend(loc='upper left')
        ax.set_title('vs Buy & Hold')
        plt.tight_layout()
        figures.savefig(key='buyhold', clear=1)
        # Drawdown image
        f, ax = self.create_ax()
        port.dly_ret_stats.plot_ltd(ax=ax)
        plt.tight_layout()
        figures.savefig(key='dd', clear=1)
        # Long / Short Returns
        f, ax = self.create_ax()
        port.plot_ret_on_dollar('B', label='All', color='k', ax=ax)
        port.long.plot_ret_on_dollar('B', label='Long', ax=ax)
        port.short.plot_ret_on_dollar('B', label='Short', ax=ax)
        ax.legend(loc='upper left')
        figures.savefig(key='ls', clear=1)
        # Sharpe / Ann Vol
        f, ax = self.create_ax()
        perf.sharpe_annualized(port.monthly_rets, expanding=1).iloc[3:].plot(ax=ax, color='k', label='sharpe')
        ax.set_ylabel('sharpe ann', color='k')
        ax2 = ax.twinx()
        perf.std_annualized(port.monthly_rets, expanding=1).iloc[3:].plot(ax=ax2, label='vol', color='b', alpha=1)
        ax2.set_ylabel('vol ann', color='b')
        plt.tight_layout()
        figures.savefig(key='sharpe', clear=1)
        # Monthly Returns Bar Chart
        f, ax = self.create_ax()
        tmp = pd.DataFrame({'All': port.monthly_rets.to_period('M'),
                            'Long': port.long.monthly_rets.to_period('M'),
                            'Short': port.short.monthly_rets.to_period('M')})
        tmp.plot(kind='bar', ax=ax, color=['k', self.long_color, self.short_color])
        AxesFormat().Y.percent().X.rotate().apply()
        plt.tight_layout()
        ax.set_title('Monthly Returns')
        figures.savefig(key='mrets', clear=1)
        # Monthly Returns Box Plot
        f, ax = self.create_ax()
        sns.boxplot(tmp, ax=ax, color=['gray', self.long_color, self.short_color])
        ax.set_title('Monthly Returns')
        AxesFormat().Y.percent().apply()
        plt.tight_layout()
        figures.savefig(key='mrets_box', clear=1)
        # Build the PDF Page
        toimg = lambda path: rlab.new_dynamic_image(path)
        itms = {
            'F1': toimg(figures['buyhold']),
            'F2': toimg(figures['dd']),
            'F3': toimg(figures['ls']),
            'F4': toimg(figures['mrets']),
            'F5': toimg(figures['sharpe']),
            'F6': toimg(figures['mrets_box']),
            'F7': stable,
            'F8': dtable,
            'HDR': self.title_bar('{0} - {1} - portfolio summary'.format(result.sid, result.desc))
        }
        pdf.build_page('portfolio', itms)


    def add_position_page(self, result):
        def dofmt(t):
            t.apply_basic_style(cmap=self.table_style)
            [row.guess_format(pcts=1, trunc_dot_zeros=1) for row in t.cells.iter_rows()]
            ncols = len(t.formatted_values.columns)
            t.set_col_widths(pcts=[1. / ncols] * ncols)
            return t

        def do_rename(df):
            d = {'consecutive_win_cnt_max': 'win_streak', 'consecutive_loss_cnt_max': 'lose_streak'}
            return df.rename(index=lambda c: d.get(c, c))

        pdf = self.pdf
        figures = self.figures
        port = result.port
        buyhold = result.buyhold

        sframe = pd.DataFrame({'all': port.positions.stats.series,
                               'long': port.long.positions.stats.series,
                               'short': port.short.positions.stats.series})
        tf = pdf.table_formatter(insert_level(sframe, 'Position', copy=True))
        stable = dofmt(tf).build()

        s = PortfolioSummary()
        s.include_long_short().include_win_loss()
        dframe = s(port, PortfolioSummary.analyze_returns).T.ix['pos']
        tf = pdf.table_formatter(do_rename(insert_level(dframe, 'Position', copy=True)))
        dtable = dofmt(tf).build()

        # Plot Position Returns
        f, ax = self.create_ax()
        port.positions.plot_rets(ax=ax)
        plt.tight_layout()
        figures.savefig(key='pos_ls', clear=1)
        # Plot Position Ranges
        f, ax = self.create_ax(figsize=(8, 3))
        port.positions.plot_ret_range(ls=1, dur=1, ax=ax)
        plt.tight_layout()
        figures.savefig(key='pos_rng', clear=1)
        # Plot Long Short Positions with regression line
        tmp = port.position_frame[['side', 'ret']].reset_index()
        g = sns.lmplot("pid", "ret", col="side", hue="side", data=tmp, size=3)
        AxesFormat().Y.percent().apply()
        figures.savefig(key='pos_ls', clear=1)
        # Plot Return vs Duration
        tmp = port.position_frame[['ret', 'duration', 'side']]
        diag_kws = {}
        if len(port.position_frame.index) <= 1:
            diag_kws = {'range': (-100, 100)}
        sns.pairplot(tmp, hue="side", size=3, diag_kws=diag_kws)
        figures.savefig(key='pos_pair', clear=1)

        toimg = lambda path: rlab.new_dynamic_image(path)
        itms = {
            'F1': toimg(figures['pos_rng']),
            'F3': toimg(figures['pos_ls']),
            'F2': toimg(figures['pos_pair']),
            'F5': stable,
            'F6': dtable,
            'HDR': self.title_bar('{0} - {1} - position summary'.format(result.sid, result.desc))
        }
        pdf.build_page('positions', itms)

