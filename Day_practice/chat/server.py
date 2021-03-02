from twisted.internet import protocol, reactor


class Chat(protocol.Protocol):
    def connectionMade(self):
        self.transport.write('connected'.encode())

    def dataReceived(self, data):
        print(data.decode('utf-8'))


class ChatFactory(protocol.Factory):
    def buildProtocol(self, addr):
        return Chat()


print('Serever started')
reactor.listenTCP(8000, ChatFactory())
reactor.run()
