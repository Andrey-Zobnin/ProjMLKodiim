import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/message.dart';

class ApiService {
  static const String _baseUrl = "http://localhost:8080";

  // GET-запрос для получения сообщения
  static Future<Message> fetchMessage() async {
    final response = await http.get(Uri.parse('$_baseUrl/api/message'));
    if (response.statusCode == 200) {
      return Message.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load message: ${response.statusCode}');
    }
  }

  // POST-запрос для отправки сообщения
  static Future<Message> sendMessage(Message message) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/api/message'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode(message.toJson()),
    );
    if (response.statusCode == 200) {
      return Message.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to send message: ${response.statusCode}');
    }
  }
}