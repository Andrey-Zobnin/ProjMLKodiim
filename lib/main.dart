import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'models/message.dart';
import 'services/api_service.dart';

void main() {
  // Установка белого фона и тёмных иконок
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarBrightness: Brightness.light,
    statusBarIconBrightness: Brightness.dark,
    systemNavigationBarColor: Colors.white,
    systemNavigationBarIconBrightness: Brightness.dark,
  ));
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'X5 ИИ-Ассистент',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E7D32),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const ChatScreen(),
    );
  }
}

// ───────────────────────── ChatScreen ──────────────────────────

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> with TickerProviderStateMixin {
  final List<ChatMessage> _messages = [];
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  bool _isComposing = false;

  void _scrollToBottom() {
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        0.0,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarBrightness: Brightness.light,
        statusBarIconBrightness: Brightness.dark,
        systemNavigationBarColor: Colors.white,
        systemNavigationBarIconBrightness: Brightness.dark,
      ),
      child: Scaffold(
        backgroundColor: Colors.white,
        appBar: _buildAppBar(context),
        body: Container(
          color: Colors.white,
          child: Stack(
            children: [
              Positioned.fill(
                child: Opacity(
                  opacity: 0.05,
                  child: Image.asset(
                    'x5.png',
                    fit: BoxFit.contain,
                    alignment: Alignment.center,
                    width: MediaQuery.of(context).size.width * 0.8,
                    height: MediaQuery.of(context).size.height * 0.8,
                  ),
                ),
              ),
              Positioned.fill(
                child: Column(
                  children: [
                    Expanded(
                      child: ListView.builder(
                        controller: _scrollController,
                        padding: const EdgeInsets.all(16),
                        reverse: true,
                        itemCount: _messages.length,
                        itemBuilder: (context, index) => AnimatedSize(
                          duration: const Duration(milliseconds: 200),
                          child: _messages[index],
                        ),
                      ),
                    ),
                    Padding(
                      padding: EdgeInsets.only(
                        bottom: MediaQuery.of(context).padding.bottom + 16,
                      ),
                      child: _buildTextComposer(context),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  PreferredSizeWidget _buildAppBar(BuildContext context) {
    return AppBar(
      systemOverlayStyle: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarBrightness: Brightness.light,
        statusBarIconBrightness: Brightness.dark,
        systemNavigationBarColor: Colors.white,
        systemNavigationBarIconBrightness: Brightness.dark,
      ),
      elevation: 0,
      backgroundColor: Colors.white,
      surfaceTintColor: Colors.white,
      title: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(8),
              boxShadow: [
                BoxShadow(
                  color: const Color.fromARGB(255, 0, 0, 0).withOpacity(0.1),
                  blurRadius: 4,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Image.asset('x5.png', fit: BoxFit.contain),
          ),
          const SizedBox(width: 12),
          const Text(
            'X5 ИИ-Ассистент',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildTextComposer(BuildContext context) {
  return Container(
    margin: const EdgeInsets.symmetric(horizontal: 16),
    padding: const EdgeInsets.symmetric(vertical: 8),
    decoration: BoxDecoration(
      color: Theme.of(context).colorScheme.surface,
      borderRadius: BorderRadius.circular(24),
      boxShadow: [
        BoxShadow(
          color: Colors.black.withOpacity(0.1),
          blurRadius: 4,
          offset: const Offset(0, -2),
        ),
      ],
    ),
    child: Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Row(
        children: [
          Flexible(
            child: TextField(
              controller: _textController,
              keyboardType: TextInputType.multiline,
              minLines: 1,        // Минимум в 1 строку
              maxLines: 5,        // Расширяется до 5 строк, затем скролл
              decoration: InputDecoration(
                hintText: 'Отправить сообщение',
                border: InputBorder.none,
                contentPadding: const EdgeInsets.symmetric(horizontal: 16),
              ),
              onChanged: (text) =>
                  setState(() => _isComposing = text.isNotEmpty),
              // при многострочном вводе onSubmitted не вызывается, но мы отправляем по кнопке
            ),
          ),
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            decoration: BoxDecoration(
              color: _isComposing
                  ? Theme.of(context).colorScheme.primary
                  : Colors.grey.withOpacity(0.3),
              shape: BoxShape.circle,
            ),
            child: IconButton(
              icon: const Icon(Icons.send, color: Colors.white),
              onPressed: _isComposing
                  ? () => _handleSubmitted(_textController.text)
                  : null,
            ),
          ),
        ],
      ),
    ),
  );
}

  void _handleSubmitted(String text) async {
    _textController.clear();
    setState(() {
      _isComposing = false;
      _messages.insert(
        0,
        ChatMessage(
          text: text,
          isUser: true,
          animationController: AnimationController(
            duration: const Duration(milliseconds: 400),
            vsync: this,
          )..forward(),
        ),
      );
    });
    _scrollToBottom();

    try {
      if (text.trim().isEmpty) {
        throw Exception('Сообщение не может быть пустым');
      }

      final message = Message(text: text.trim());
      final response = await ApiService.sendMessage(message);
      
      if (response == null) {
        throw Exception('Не удалось получить ответ от сервера');
      }
      
      if (response.text.trim().isEmpty) {
        throw Exception('Получен пустой ответ от сервера');
      }
      
      setState(() {
        _messages.insert(
          0,
          ChatMessage(
            text: response.text,
            isUser: false,
            animationController: AnimationController(
              duration: const Duration(milliseconds: 400),
              vsync: this,
            )..forward(),
          ),
        );
      });
      _scrollToBottom();
    } catch (e) {
      // Показываем ошибку пользователю
      final errorMessage = e.toString().replaceAll('Exception: ', '');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(errorMessage),
          backgroundColor: Theme.of(context).colorScheme.error,
          behavior: SnackBarBehavior.floating,
          margin: EdgeInsets.only(
            bottom: 80,
            right: 20,
            left: 20,
          ),
        ),
      );
    }
  }
}

// ───────────────────────── ChatMessage ──────────────────────────

class ChatMessage extends StatelessWidget {
  ChatMessage({
    super.key,
    required this.text,
    required this.isUser,
    required this.animationController,
  }) : timestamp = DateTime.now();

  final String text;
  final bool isUser;
  final AnimationController animationController;
  final DateTime timestamp;

  @override
  Widget build(BuildContext context) {
    return SizeTransition(
      sizeFactor: CurvedAnimation(
        parent: animationController,
        curve: Curves.easeOutQuad,
      ),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 10),
        child: Row(
          mainAxisAlignment:
              isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (!isUser) _buildAvatar(context, false),
            Flexible(child: _buildBubble(context)),
            if (isUser) _buildAvatar(context, true),
          ],
        ),
      ),
    );
  }

  Widget _buildAvatar(BuildContext context, bool forUser) {
    final colorScheme = Theme.of(context).colorScheme;
    return Container(
      margin: EdgeInsets.only(right: forUser ? 0 : 16, left: forUser ? 16 : 0),
      padding: const EdgeInsets.all(2),
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: LinearGradient(
          colors: [
            forUser ? colorScheme.primary : colorScheme.secondary,
            forUser
                ? colorScheme.primary.withOpacity(0.7)
                : colorScheme.secondary.withOpacity(0.7),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: (forUser ? colorScheme.primary : colorScheme.secondary)
                .withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: CircleAvatar(
        backgroundColor: Colors.transparent,
        child: Icon(
          forUser ? Icons.person : Icons.smart_toy,
          color: Colors.white,
        ),
      ),
    );
  }

  Widget _buildBubble(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      constraints: BoxConstraints(
        maxWidth: MediaQuery.of(context).size.width * 0.75, // Ограничение ширины сообщения
      ),
      // Удаляем фиксированную ширину, чтобы текст мог переноситься
      decoration: BoxDecoration(
        color: const Color.fromARGB(255, 184, 233, 204),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
        borderRadius: BorderRadius.only(
          topLeft: const Radius.circular(16),
          topRight: const Radius.circular(16),
          bottomLeft: Radius.circular(isUser ? 16 : 0),
          bottomRight: Radius.circular(isUser ? 0 : 16),
        ),
      ),
      child: Column(
        crossAxisAlignment:
            isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
        children: [
          Text(
            isUser ? 'Вы' : 'X5 Ассистент',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 14,
              color: isUser ? colorScheme.primary : colorScheme.secondary,
            ),
          ),
          const SizedBox(height: 5),
          // Обрабатываем текст сообщения, разбивая его на строки
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: text.split('\n').map((line) => 
              Padding(
                padding: const EdgeInsets.only(bottom: 2),
                child: Text(
                  line,
                  style: TextStyle(
                    fontSize: 16,
                    height: 1.4,
                    color: colorScheme.onSurface,
                  ),
                  softWrap: true,
                  overflow: TextOverflow.clip,
                  textAlign: TextAlign.left,
                  maxLines: null, // Разрешаем неограниченное количество строк
                  textWidthBasis: TextWidthBasis.longestLine, // Используем самую длинную строку как основу для ширины
                ),
              )
            ).toList(),
          ),
          const SizedBox(height: 4),
          Text(
            "${timestamp.hour.toString().padLeft(2, '0')}:${timestamp.minute.toString().padLeft(2, '0')}",
            style: TextStyle(
              fontSize: 12,
              color: colorScheme.onSurface.withOpacity(0.6),
            ),
          ),
        ],
      ),
    );
  }
}
