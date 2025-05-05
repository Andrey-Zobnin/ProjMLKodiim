import 'package:http/http.dart' as http;

import 'dart:convert';
import '../models/message.dart';

class ApiService {
  static const String _baseUrl = "http://localhost:8080";

  // GET-запрос для получения сообщения
  static Future<Message> fetchMessage({String? token}) async {
    final headers = <String, String>{};
    if (token != null) {
      headers['Authorization'] = 'Bearer $token';
    }

    final response = await http.get(
      Uri.parse('$_baseUrl/api/message'),
      headers: headers,
    );

    if (response.statusCode == 200) {
      return Message.fromJson(json.decode(response.body));
    } else if (response.statusCode == 401) {
      throw Exception('Unauthorized: Invalid or missing token');
    } else {
      throw Exception('Failed to load message: ${response.statusCode} ${response.reasonPhrase}');
    }
  }

  // POST-запрос для отправки сообщения
  static Future<Message> sendMessage(Message message, {String? token}) async {
    final headers = <String, String>{
      'Content-Type': 'application/json',
    };
    if (token != null) {
      headers['Authorization'] = 'Bearer $token';
    }

    final response = await http.post(
      Uri.parse('$_baseUrl/api/message'),
      headers: headers,
      body: json.encode(message.toJson()),
    );

    if (response.statusCode == 200) {
      try {
        final jsonResponse = json.decode(response.body);
        if (jsonResponse == null) {
          throw Exception('Получен пустой ответ от сервера');
        }
        return Message.fromJson(jsonResponse);
      } catch (e) {
        throw Exception('Ошибка при обработке ответа сервера: ${e.toString()}');
      }
    } else if (response.statusCode == 401) {
      throw Exception('Unauthorized: Invalid or missing token');
    } else {
      throw Exception('Failed to send message: ${response.statusCode} ${response.reasonPhrase}');
    }
  }
}