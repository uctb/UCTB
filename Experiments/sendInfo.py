# coding=utf-8

import smtplib
from email.mime.text import MIMEText
from email.header import Header
import socket
from_addr = '815128340@qq.com'  # 邮件发送账号
to_addrs = '815128340@qq.com'  # 接收邮件账号
AuthCode = 'dlchcdksvvwsbejb'  # 授权码（这个要填自己获取到的）
#dlchcdksvvwsbejb
smtp_server = 'smtp.qq.com'
smtp_port = 465  # 固定端口


def senInfo(info):
    smtp = smtplib.SMTP_SSL(smtp_server, smtp_port)
    #smtp.set_debuglevel(1)
    smtp.login(from_addr, AuthCode)
    # 组装发送内容
    message = MIMEText('{} 机器上的任务已完成，CodeVersion: {}'.format(socket.gethostname(),info), 'plain', 'utf-8')  # 发送的内容
    message['From'] = from_addr  # 发件人
    message['To'] = to_addrs  # 收件人
    subject = '任务已完成'
    message['Subject'] = Header(subject, 'utf-8')  # 邮件标题

    try:
        # 配置服务器
        smtp.sendmail(from_addr, to_addrs, message.as_string())

    except Exception as e:
        print('邮件发送失败' + str(e))
    smtp.close()
    return True


if __name__ == "__main__":
    senInfo("{}, 请尽快查看~".format("1号任务已完成"))