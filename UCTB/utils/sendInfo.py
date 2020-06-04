# coding=utf-8

import smtplib
from email.mime.text import MIMEText
from email.header import Header
import socket
from_addr = '815128340@qq.com'  # send account
to_addrs = '815128340@qq.com'  # receive account
AuthCode = 'dlchcdksvvwsbejb' 
smtp_server = 'smtp.qq.com'
smtp_port = 465  # port


def senInfo(info):
    smtp = smtplib.SMTP_SSL(smtp_server, smtp_port)
    #smtp.set_debuglevel(1)
    smtp.login(from_addr, AuthCode)
    # assemble message
    message = MIMEText('The task on Machine {} has been done, CodeVersion: {}'.format(socket.gethostname(),info), 'plain', 'utf-8')  # content
    message['From'] = from_addr  # sender
    message['To'] = to_addrs  # receiver
    subject = 'Task Done'
    message['Subject'] = Header(subject, 'utf-8')  # email title

    try:
        smtp.sendmail(from_addr, to_addrs, message.as_string())

    except Exception as e:
        print('Fail to send mail.' + str(e))
    smtp.close()
    return True


if __name__ == "__main__":
    senInfo("{}, Check it~".format("Task 1 has been done."))