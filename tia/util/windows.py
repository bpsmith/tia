"""
collection of utilities for use on windows systems
"""
import os


def send_outlook_email(to, subject, body, attachments=None, cc=None, bcc=None, is_html=0):
    """ Send an email using your local outlook client """
    import win32com.client
    asarr = lambda v: None if not v else isinstance(v, str) and [v] or v
    def update_recipients(robj, users, type):
        users = asarr(to)
        if users:
            for u in users:
                r = robj.Add(u)
                r.Type = type

    outlook = win32com.client.gencache.EnsureDispatch("Outlook.Application")
    mapi = outlook.GetNamespace("MAPI")
    constants = win32com.client.constants
    msg = outlook.CreateItem(0)
    # setup the recipients
    recipients = msg.Recipients
    to and update_recipients(recipients, to, constants.olTo)
    cc and update_recipients(recipients, cc, constants.olCC)
    bcc and update_recipients(recipients, bcc, constants.olBCC)
    recipients.ResolveAll()
    msg.Subject = subject
    if is_html:
        msg.BodyFormat = constants.olFormatHTML
        msg.HTMLBody = body
    else:
        msg.Body = body

    list(map(lambda fpath: msg.Attachments.Add(fpath), attachments or []))
    msg.Send()


class WinSCPBatch(object):
    """ Provide a utility class which invokes the Winscp processes via the command line.

    Example
    -------
    batch = WinSCPBatch('your_session_name', logfile='c:\\temp\\winscp\\mylog.log')
    batch.add_download('remotefile.txt', 'c:\\temp\\winscp\\localfile.txt')
    batch.execute()
    """
    def __init__(self, session, logfile=None):
        self.session = session
        self.logfile = logfile
        self.cmds = []
        self.double_quote = lambda s: s and '""' + s + '""' or ''

    def add_download(self, remote, local):
        cmd = 'get %s %s' % (self.double_quote(remote), self.double_quote(local))
        self.cmds.append(cmd)

    def add_downloads(self, filemap):
        """Add the dict of downloads. (Note the Winscp command line accepts wildcards)

        Parameters
        ----------
        filemap: dict, (remote_filename -> local_filename)
        """
        [self.add_download(k, v) for k, v in filemap.items()]

    def add_upload(self, remote, local):
        cmd = 'put %s %s' % (self.double_quote(local), self.double_quote(remote))
        self.cmds.append(cmd)

    def add_uploads(self, filemap):
        """Add the dict of uploads

        Parameters
        ----------
        filemap: dict, (remote_filename -> local_filename)
        """
        [self.add_upload(k, v) for k, v in filemap.items()]

    def add_cd(self, remote_dir):
        cmd = 'cd %s' % remote_dir
        self.cmds.append(cmd)

    def execute(self):
        env = os.environ['PATH']
        if 'WinSCP' not in env:
            if os.path.exists('C:\Program Files (x86)\WinSCP'):
                os.environ['PATH'] = env + ';C:\Program Files (x86)\WinSCP'
            elif os.path.exists('C:\Program Files\WinSCP'):
                os.environ['PATH'] = env + ';C:\Program Files\WinSCP'

        cmd = 'winscp.exe'
        if self.logfile:
            cmd += ' /log="%s"' % self.logfile

        cmd += ' /command'
        cmd += ' "option batch abort"'
        cmd += ' "option confirm off"'
        cmd += ' "open %s"' % self.session
        for c in self.cmds:
            cmd += ' "%s"' % c.strip()

        cmd += ' "exit"'
        # not able to detect failures - but can raise failures when looking for expected files
        os.system(cmd)
        #import subprocess as sub
        #p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
        #output, errors = p.communicate()
        #print output




