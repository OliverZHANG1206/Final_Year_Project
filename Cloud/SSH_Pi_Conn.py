import paramiko

def get_file_from_Pi():
    # Create an SSH client
    ssh_client = paramiko.SSHClient()

    # Automatically add the server's host key (this is insecure in production)
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the Raspberry Pi
    ssh_client.connect(hostname='192.168.3.98', username='david', password='123456')

    # Create an SFTP client
    sftp_client = ssh_client.open_sftp()

    # Download the file from the Raspberry Pi
    local_path = './Data/Data.db'
    remote_path = '/home/david/Desktop/Hub/Data.db'
    sftp_client.get(remote_path, local_path)

    # Close the SFTP session and SSH connection
    sftp_client.close()
    ssh_client.close()


